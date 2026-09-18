"""Repair only Ninja 1.11.1.1's misplaced wheel Tag headers, not its binary."""
import base64
import csv
from email.parser import Parser
import hashlib
import importlib.metadata
import io
from pathlib import Path


def repair_metadata(info: Path) -> bool:
    metadata = Parser().parsestr((info / "METADATA").read_text())
    if metadata.get("Name") != "ninja" or metadata.get("Version") != "1.11.1.1":
        raise RuntimeError("refusing to repair a different Ninja distribution")
    wheel = info / "WHEEL"
    original = wheel.read_text()
    if Parser().parsestr(original).get_all("Tag"):
        return False
    headers, separator, body = original.partition("\n\n")
    if not separator or not body.strip() or any(not line.startswith("Tag: ") for line in body.strip().splitlines()):
        raise RuntimeError("unexpected Ninja wheel metadata structure")
    repaired = headers + "\n" + body
    if not Parser().parsestr(repaired).get_all("Tag"):
        raise RuntimeError("Ninja wheel tag repair did not restore headers")
    payload = repaired.encode("utf-8")
    record = info / "RECORD"
    with record.open(newline="") as stream:
        rows = list(csv.reader(stream))
    matches = [row for row in rows if row[0] == info.name + "/WHEEL"]
    if len(matches) != 1:
        raise RuntimeError("Ninja RECORD must identify exactly one WHEEL file")
    matches[0][1:] = ["sha256=" + base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode().rstrip("="), str(len(payload))]
    contents = io.StringIO(newline="")
    csv.writer(contents).writerows(rows)
    wheel.write_bytes(payload)
    record.write_text(contents.getvalue())
    return True


if __name__ == "__main__":
    distribution = importlib.metadata.distribution("ninja")
    wheel_files = [entry for entry in distribution.files if entry.name == "WHEEL" and entry.parent.name == "ninja-1.11.1.1.dist-info"]
    if len(wheel_files) != 1:
        raise RuntimeError("unexpected installed Ninja wheel location")
    info = Path(distribution.locate_file(wheel_files[0])).parent
    changed = repair_metadata(info)
    print(f"Ninja 1.11.1.1 wheel metadata repaired={changed}; executable/version unchanged", flush=True)
