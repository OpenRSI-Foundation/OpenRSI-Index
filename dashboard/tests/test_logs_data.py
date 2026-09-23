"""Archive correctness contracts, using tiny committed repositories."""
import gzip
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('logs_build', ROOT / 'tools/build.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)
from parsers import atif, codex_lines, jsonl_lines


class LogsDataTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True)

    def put(self, path, data):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data) if not isinstance(data, str) else data)
        return target

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')

    def export(self):
        return build.build(self.repo, self.root / 'output')

    def events(self, run):
        folder = self.root / 'output' / run['id']
        index = json.loads((folder / 'index.json').read_text())
        return [event for page in index['pages'] for event in json.loads(gzip.decompress((folder / page['file']).read_bytes()))]

    def test_gzip_resumes_source_pinning_and_timeout_score(self):
        base = 'rsi-logs/task/codex-gpt-5.6-sol/'
        self.put(base + 'evolve_state.json', {'best_score': 0, 'submissions': [{'round':'agent-1', 'score':0,'status':'completed'}]})
        self.put(base + 'final_result.json', {'status':'failed','timed_out':True,'best_score':0,'agent':'codex','model':'gpt-5.6-sol'})
        compressed = self.repo / base / 'agent_output.txt.gz'
        compressed.write_bytes(gzip.compress(b'OpenAI Codex\nuser\nstart\ncodex\nfirst\n'))
        self.put(base + 'agent_output.resume-10.txt','codex\ntenth\n')
        self.put(base + 'agent_output.resume-2.txt','codex\nsecond\n')
        self.commit()
        self.put(base + 'agent_output.resume-3.txt','codex\nuntracked private text\n')
        result = self.export(); run = result['runs'][0]
        self.assertEqual(run['status'], 'timed_out')
        self.assertEqual(run['best_score'], 0)
        self.assertIn(result['commit'], run['source_url'])
        messages = [e['text'] for e in self.events(run) if e['kind'] == 'assistant']
        self.assertEqual(messages, ['first','second','tenth'])

    def test_missing_final_is_not_running_and_untracked_metadata_is_excluded(self):
        base = 'rsi-logs/task/codex-gpt-5.6-sol/'
        self.put(base + 'evolve_state.json', {'best_score':1,'submissions':[]})
        self.commit()
        self.put(base + 'final_result.json', {'status':'completed','model':'private metadata'})
        run = self.export()['runs'][0]
        self.assertEqual(run['status'],'archived')
        self.assertNotEqual(run['model'],'private metadata')
        self.assertEqual(run['events'],0)
        self.assertEqual(run['format'],'Transcript')

    def test_modified_archive_cannot_claim_committed_provenance(self):
        path = self.put('rsi-logs/task/run/evolve_state.json',{})
        self.commit(); path.write_text('{"best_score":42}')
        with self.assertRaisesRegex(ValueError, 'Commit changes'):
            self.export()

    def test_long_events_are_complete_and_pages_are_bounded(self):
        body = 'UTF-8: 研究 <script>alert(1)</script>\n' * 15000
        result = build.publish_events([{'kind':'tool','title':'Bash','text':body}], self.root/'pages')
        events = []
        for page in result['pages']:
            raw = gzip.decompress((self.root/'pages'/page['file']).read_bytes())
            self.assertLess(len(raw), 250000)
            events.extend(json.loads(raw))
        self.assertEqual(''.join(e['text'] for e in events),body)
        self.assertEqual(result['tools'],1)
        self.assertGreater(len(result['pages']),1)

    def test_gpic_different_protocols_never_become_a_score_curve(self):
        self.put('rsi-logs/signature-tasks/gpic-10m-autoresearch/codex-gpt-5.6-sol/experiments.jsonl',
                 '{"attempt_id":"probe","fd_val_screening":2,"status":"completed"}\n')
        self.commit(); run = self.export()['runs'][0]
        self.assertEqual(run['submissions'],[])
        self.assertIsNone(run['best_score'])
        self.assertEqual(self.events(run)[0]['title'],'probe')

    def test_codex_markers_inside_fences_stay_in_the_message(self):
        rows = list(codex_lines('codex\nExample:\n```\nuser\nthinking\n```\nexec\nls\n'.splitlines(True)))
        self.assertEqual([r['kind'] for r in rows],['assistant','tool'])
        self.assertIn('user\nthinking',rows[0]['text'])

    def test_claude_preamble_reasoning_tool_results_and_subagents_survive(self):
        rows = ['Container diagnostic\n', json.dumps({'type':'assistant','parent_tool_use_id':'parent','message':{'content':[
            {'type':'thinking','thinking':'reason'}, {'type':'tool_use','id':'call','name':'Bash','input':{'command':'ls'}}]}}),
            json.dumps({'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'call','content':'output','is_error':True}]}})]
        events = list(jsonl_lines(rows))
        self.assertEqual([e['kind'] for e in events],['system','thinking','tool','result'])
        self.assertEqual(events[2]['parent'],'parent')
        self.assertEqual(events[2]['call_id'],events[3]['call_id'])
        self.assertTrue(events[3]['error'])

    def test_atif_message_tools_and_observations_remain_in_order(self):
        rows = list(atif({'steps':[{'step_id':1,'source':'agent','message':'hello','tool_calls':[
            {'function_name':'Read','tool_call_id':'a','arguments':{'path':'x'}}],
            'observation':{'results':[{'source_call_id':'a','content':'contents'}]}}]}))
        self.assertEqual([r['kind'] for r in rows],['assistant','tool','result'])
        self.assertEqual(rows[-1]['text'],'contents')


if __name__ == '__main__':
    unittest.main()
