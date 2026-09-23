"""End-to-end archive navigation, content safety, and responsive layout."""
import gzip
import http.server
import json
import os
from pathlib import Path
import threading
import unittest

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


class LogsBrowserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        manifest = ROOT / 'assets/logs-data/index.json'
        if not manifest.exists():
            raise unittest.SkipTest('Build logs data before running browser tests.')
        cls.data = json.loads(manifest.read_text())
        cls.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
            lambda *args, **kwargs: QuietHandler(*args, directory=ROOT, **kwargs))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.pw = sync_playwright().start()
        options = {'headless':True}
        if os.environ.get('PLAYWRIGHT_CHROMIUM_EXECUTABLE'):
            options['executable_path'] = os.environ['PLAYWRIGHT_CHROMIUM_EXECUTABLE']
        cls.browser = cls.pw.chromium.launch(**options)
        cls.url = f'http://127.0.0.1:{cls.server.server_port}/index.html'

    @classmethod
    def tearDownClass(cls):
        cls.browser.close(); cls.pw.stop(); cls.server.shutdown(); cls.server.server_close()
        cls.thread.join(timeout=2)

    def setUp(self):
        self.page = self.browser.new_page(viewport={'width':1440,'height':1050})
        self.addCleanup(self.page.close)
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        self.page.goto(self.url)
        expect(self.page.locator('#runs .run-row')).to_have_count(len(self.data['runs']))

    def test_filters_search_reset_and_empty_state(self):
        self.page.get_by_role('button',name='Signature tasks',exact=True).click()
        expect(self.page.locator('#runs .run-row')).to_have_count(sum(r['track']=='Signature' for r in self.data['runs']))
        self.page.locator('#search').fill('impossible-no-such-task')
        expect(self.page.locator('#empty-filter')).to_be_visible()
        self.page.locator('#empty-reset').click()
        expect(self.page.locator('#runs .run-row')).to_have_count(len(self.data['runs']))
        self.page.locator('#agent').select_option('codex')
        self.page.locator('#search').fill('Molmo2')
        expect(self.page.locator('#runs .run-row')).to_have_count(1)
        self.assertEqual(self.errors,[])

    def test_tasks_group_models_with_working_expand_collapse_and_sort(self):
        tasks = {r['task'] for r in self.data['runs']}
        expect(self.page.locator('.task-group')).to_have_count(len(tasks))
        expect(self.page.locator('.task-group[open]')).to_have_count(0)
        for task in tasks:
            group = self.page.locator(f'.task-group[data-task="{task}"]')
            expect(group.locator('summary .task-name')).to_have_count(1)
            expect(group.locator('.run-row')).to_have_count(sum(r['task']==task for r in self.data['runs']))
        self.page.get_by_role('button',name='Expand all',exact=True).click()
        expect(self.page.locator('.task-group[open]')).to_have_count(len(tasks))
        self.page.locator('#task-sort').select_option('events')
        expected = max(tasks,key=lambda task: sum(r['events'] for r in self.data['runs'] if r['task']==task))
        expect(self.page.locator('.task-group').first).to_have_attribute('data-task',expected)
        self.page.get_by_role('button',name='Collapse all',exact=True).click()
        expect(self.page.locator('.task-group[open]')).to_have_count(0)
        self.page.locator('#search').fill('Molmo2')
        expect(self.page.locator('.task-group')).to_have_count(1)
        expect(self.page.locator('.run-row')).to_have_count(2)

    def test_compressed_trace_pagination_submissions_artifacts_and_back(self):
        run = next(r for r in self.data['runs'] if r['task']=='liger-tied-ce' and r['agent']=='codex')
        self.page.goto(self.url + '#run=' + run['id'])
        expect(self.page.locator('.events .event').first).to_be_visible()
        first = self.page.locator('.event-id').first.inner_text()
        self.page.locator('.pager .next').first.click()
        expect(self.page).to_have_url(self.url + '#run=' + run['id'] + '&tab=trajectory&page=2')
        expect(self.page.get_by_role('spinbutton',name='Page number',exact=True)).to_have_value('2')
        expect(self.page.locator('.events .event').first).to_be_visible()
        expect(self.page.locator('.event-id').first).not_to_have_text(first)
        self.page.reload()
        expect(self.page.get_by_role('spinbutton',name='Page number',exact=True)).to_have_value('2')
        expect(self.page.locator('.events .event').first).to_be_visible()
        details = self.page.locator('.events details').first
        details.locator('summary').click()
        expect(details.locator('pre, .prose').first).to_be_visible()
        self.page.get_by_role('link',name='Submissions',exact=True).click()
        expect(self.page.locator('.submissions tbody tr')).to_have_count(len(run['submissions']))
        self.page.get_by_role('link',name='Artifacts',exact=True).click()
        for link in self.page.locator('.artifact-list a').all():
            self.assertIn(self.data['commit'],link.get_attribute('href'))
        self.page.get_by_role('link',name='All research logs').click()
        expect(self.page.locator('#overview')).to_be_visible()
        self.assertEqual(self.errors,[])

    def test_log_markup_is_inert_and_failure_has_a_retry(self):
        run = self.data['runs'][0]
        row = {'id':1,'kind':'assistant','title':'<img src=x onerror=alert(1)>',
               'text':'# A heading\n\n<script>window.pwned=1</script>\n\n**Safe bold** and `code`\n\n```html\n<img src=x onerror=alert(1)>\n```'}
        pattern = '**/assets/logs-data/' + run['id'] + '/*.json.gz*'
        self.page.route(pattern, lambda route: route.fulfill(body=gzip.compress(json.dumps([row]).encode()),content_type='application/gzip'))
        self.page.goto(self.url + '#run=' + run['id'])
        expect(self.page.locator('.events h3')).to_have_text('A heading')
        expect(self.page.locator('.events strong').last).to_have_text('Safe bold')
        self.assertEqual(self.page.locator('.events script, .events img').count(),0)
        self.assertIsNone(self.page.evaluate('window.pwned'))
        self.page.unroute(pattern)
        self.page.route(pattern, lambda route: route.fulfill(status=503,body='Unavailable'))
        self.page.reload()
        expect(self.page.locator('#tab-content [role=alert]')).to_be_visible()
        self.page.unroute(pattern)
        self.page.get_by_role('button',name='Try again',exact=True).click()
        expect(self.page.locator('.events .event').first).to_be_visible()

    def test_mobile_and_desktop_have_no_page_overflow(self):
        for width in [390,768,1440]:
            self.page.set_viewport_size({'width':width,'height':900})
            self.assertLessEqual(self.page.evaluate('document.documentElement.scrollWidth'),width)
        self.page.locator('.task-group summary').first.click()
        self.page.locator('.row-link').first.click()
        expect(self.page.locator('.events .event').first).to_be_visible()
        for width in [390,768,1440]:
            self.page.set_viewport_size({'width':width,'height':900})
            self.assertLessEqual(self.page.evaluate('document.documentElement.scrollWidth'),width)

    def test_all_archive_formats_open_in_the_standalone_dashboard(self):
        for form in ['ATIF trajectory','Experiment journal','Research report']:
            run = next(r for r in self.data['runs'] if r['format']==form)
            self.page.goto(self.url + '#run=' + run['id'])
            expect(self.page.locator('.events .event').first).to_be_visible()
            self.assertEqual(self.page.locator('.chart').count(),0)
        self.page.get_by_role('link',name='Dashboard',exact=True).click()
        expect(self.page.locator('#overview')).to_be_visible()
        self.assertEqual(self.errors,[])


if __name__ == '__main__':
    unittest.main()
