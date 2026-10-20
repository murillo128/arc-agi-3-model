"""Restored monitoring contracts; HTTP and data checks, never visual tests."""
import json
from html.parser import HTMLParser
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen

from model.dashboard import make_server


class PageStructure(HTMLParser):
    """Read ownership of native controls without rendering the page."""
    def __init__(self, html):
        super().__init__()
        self.stack = []
        self.nodes = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        node = dict(tag=tag, attrs=dict(attrs), parents=list(self.stack))
        self.nodes.append(node)
        if tag not in {'meta', 'link', 'input', 'br', 'hr', 'img', 'source'}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]['tag'] == tag:
                del self.stack[index:]
                break


class DashboardRestorationTests(unittest.TestCase):
    def test_shared_english_shell_keeps_local_tabs_and_native_links_in_one_navigation(self):
        assets = Path(__file__).parents[1] / 'dashboard_web'
        for name, sections in (
            ('sequences', ('world', 'path', 'evaluation', 'curriculum')),
            ('optimization', ('pulse', 'path', 'memory')),
            ('subgoals', ('world', 'path', 'evaluation', 'curriculum')),
            ('index', ('world', 'behavior', 'evaluations')),
            ('campaign', ()),
        ):
            with self.subTest(page=name):
                page = PageStructure((assets / f'{name}.html').read_text())
                html = next(n for n in page.nodes if n['tag'] == 'html')
                self.assertEqual(html['attrs']['lang'], 'en')
                navs = [n for n in page.nodes if n['tag'] == 'nav']
                self.assertEqual(len(navs), 1)
                nav = navs[0]
                self.assertEqual(nav['attrs']['class'], 'dashboard-tabs')
                self.assertTrue(any(p['attrs'].get('class') == 'dashboard-header' for p in nav['parents']))
                ids = [n['attrs']['id'] for n in page.nodes if 'id' in n['attrs']]
                self.assertEqual(len(ids), len(set(ids)), 'every control has a unique target')
                for section in sections:
                    tab = next(n for n in page.nodes if n['attrs'].get('id') == f'tab-{section}')
                    self.assertEqual(tab['tag'], 'button')
                    self.assertEqual(tab['attrs']['role'], 'tab')
                    self.assertIn(nav, tab['parents'])
                    self.assertIn(tab['attrs']['aria-controls'], ids)
                links = [n for n in page.nodes if n['tag'] == 'a' and nav in n['parents']]
                self.assertTrue(links)
                self.assertTrue(all(n['attrs'].get('href') for n in links))
                self.assertTrue(all(n['attrs'].get('role') != 'tab' for n in links))
                if name == 'sequences':
                    by_id = {n['attrs']['id']: n for n in page.nodes if 'id' in n['attrs']}
                    games = by_id['curriculum-games']
                    self.assertIn(by_id['curriculum'], games['parents'])
                    self.assertFalse(any(p['attrs'].get('class') == 'dashboard-header' for p in games['parents']))
                    for key in ('configuration', 'learning-rates', 'run-info'):
                        parents = by_id[key]['parents']
                        self.assertTrue(any(p['tag'] == 'details' for p in parents))
                        self.assertFalse(any(p['attrs'].get('class') == 'dashboard-header' for p in parents))
                    self.assertEqual(by_id['history-link']['attrs']['href'], 'history.html#path')

    def test_sequence_home_retains_monitoring_sections_alongside_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'report.json'
            report.write_text(json.dumps(dict(kind='game_sequences', history=[])))
            server = make_server(('127.0.0.1', 0), None, sequence_report=report)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urlopen(f'http://127.0.0.1:{server.server_port}/') as response:
                    html = response.read().decode()
                for section in ('representation-loss', 'predictor-components', 'latent-diversity',
                                'latent-rank', 'latent-inactive', 'latent-sensitivity', 'changes',
                                'fixed-quality', 'tab-path', 'tab-evaluation', 'tab-curriculum'):
                    with self.subTest(section=section):
                        self.assertIn(f'id="{section}"', html)
                self.assertIn('optimization.html', html)
                self.assertLess(html.index('src="chart-tools.js"'), html.index('src="sequences.js"'))
            finally:
                server.shutdown(); server.server_close(); thread.join()

    def test_archived_run_and_trajectory_are_read_only_and_never_replace_current_metrics(self):
        from urllib.error import HTTPError
        from urllib.request import Request
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current, archive = root/'report.json', root/'old.json'
            current.write_text(json.dumps(dict(kind='game_sequences', status='running', progress=dict(actions=16))))
            archive.write_text(json.dumps(dict(kind='synthetic_subgoals', status='running',
                progress=dict(actions=999), planner_trajectory=dict(id='old-trajectory'))))
            (root/'planner_trajectory.json').write_text(json.dumps(dict(kind='planner_trajectory', id='old-trajectory')))
            server = make_server(('127.0.0.1', 0), None, sequence_report=current, history_subgoal_report=archive)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f'http://127.0.0.1:{server.server_port}'
                def read(path):
                    with urlopen(url+path) as response:
                        return json.load(response)
                state = read('/api/state')
                self.assertTrue(state['history_available'])
                self.assertEqual(state['sequences']['progress']['actions'], 16)
                self.assertNotIn('subgoals', state)
                self.assertNotIn('curriculum_control', state)
                history = read('/api/history/state')
                self.assertEqual(history['subgoals']['progress']['actions'], 999)
                self.assertEqual(history['subgoals']['status'], 'historical')
                self.assertTrue(history['subgoals']['historical'])
                self.assertIsNone(history['curriculum_control'])
                self.assertNotIn('sequences', history)
                self.assertEqual(read('/api/history/trajectory')['trajectory']['id'], 'old-trajectory')
                with urlopen(url+'/history.html#path') as response:
                    html = response.read().decode()
                for section in ('planner-success', 'path-loss', 'efficiency', 'reward', 'subgoal-outcomes', 'trajectory-info'):
                    self.assertIn(f'id="{section}"', html)
                self.assertLess(html.index('src="chart-tools.js"'), html.index('src="subgoals.js"'))
                for path in ('/api/curriculum', '/api/history/curriculum'):
                    with self.subTest(path=path), self.assertRaises(HTTPError) as failure:
                        urlopen(Request(url+path, data=b'{"operation":"level"}', headers={'Content-Type':'application/json'}))
                    self.assertEqual(failure.exception.code, 404)
                # Missing/corrupt archived data does not break the current run.
                archive.unlink()
                self.assertEqual(read('/api/history/state')['subgoals']['status'], 'waiting')
                self.assertIsNone(read('/api/history/trajectory')['trajectory'])
                archive.write_text('[]')
                self.assertEqual(read('/api/history/state')['subgoals']['status'], 'waiting')
                self.assertEqual(read('/api/state')['sequences']['progress']['actions'], 16)
            finally:
                server.shutdown(); server.server_close(); thread.join()
