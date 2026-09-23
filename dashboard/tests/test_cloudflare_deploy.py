"""Deployment boundaries: only static dashboard files and its dedicated Worker."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("deploy_cloudflare", ROOT / "tools/deploy_cloudflare.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class CloudflareDeployTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in deploy.STATIC_FILES:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        self.data = self.root / "assets/logs-data"
        self.run = self.data / ("a" * 16)
        self.run.mkdir(parents=True)
        (self.data / "index.json").write_text(json.dumps({"runs":[{"id":"a"*16}], "commit":"b"*40}))
        (self.run / "index.json").write_text(json.dumps({"pages":[{"file":"0000.json.gz"}]}))
        (self.run / "0000.json.gz").write_bytes(b"trace")

    def test_only_referenced_public_assets_are_uploaded(self):
        (self.root / "credentials.json").write_text("private")
        (self.root / "serve.py").write_text("source code")
        (self.run / "9999.json.gz").write_bytes(b"stale private page")
        _, paths = deploy.dashboard_files(self.root)
        names = {path.relative_to(self.root).as_posix() for path in paths}
        self.assertNotIn("credentials.json", names)
        self.assertNotIn("serve.py", names)
        self.assertNotIn("assets/logs-data/" + "a"*16 + "/9999.json.gz", names)
        self.assertEqual(len(paths), len(deploy.STATIC_FILES) + 3)

    def test_manifest_cannot_reference_files_outside_its_run(self):
        (self.run / "index.json").write_text(json.dumps({"pages":[{"file":"../../../credentials.json"}]}))
        with self.assertRaisesRegex(ValueError, "Invalid trace page"):
            deploy.dashboard_files(self.root)

    def test_existing_unrelated_worker_is_never_overwritten(self):
        credentials = self.root / "credentials.json"
        credentials.write_text(json.dumps({"api_token":"private-token", "account_id":"c"*32,
                                           "worker_name":"openrsi-handoff"}))
        requests = []
        def handle(request):
            requests.append((request.method, request.url.path))
            return httpx.Response(200, json={"success":True,"result":{"tags":["another-app"]}})
        client = httpx.Client(transport=httpx.MockTransport(handle))
        index, paths = deploy.dashboard_files(self.root)
        with patch.object(deploy, "dashboard_files", return_value=(index, paths)), \
             patch.object(deploy, "manifest", return_value=deploy.manifest(paths,self.root)), \
             patch.object(deploy.httpx, "Client", return_value=client), \
             patch.dict(deploy.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                deploy.deploy(credentials)
        self.assertEqual(requests, [("GET", "/client/v4/accounts/" + "c"*32 + "/workers/scripts/openrsi-logs/settings")])


if __name__ == "__main__":
    unittest.main()
