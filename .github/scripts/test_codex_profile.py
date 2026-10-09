"""Offline regressions for issue settings, project assignment and both launchers.

Run: python3 -m unittest discover -s .github/scripts -p 'test_codex_profile.py'
No model turn, host config, GitHub mutation or product suite is used.
"""

import ast
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from codex_profile import CAP_KEY, ProfilePolicy, issue_settings, load_settings, prepare_issue
from prepare_pr_audit import build_audit_client


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / '.github/workflows/codex-execute-ready.yml').read_text()
AUDIT = (ROOT / '.github/workflows/codex-review-ready.yml').read_text()
BEGIN = '          cat > "$client" <<\'PY\'\n'
EXECUTOR = textwrap.dedent(WORKFLOW.split(BEGIN, 1)[1].split('\n          PY\n', 1)[0])


class Server:
    def __init__(self, home):
        self.home, self.calls = home, []
        self.saved = {'model': 'gpt-6-astra', 'reasoningEffort': 'xhigh'}
        self.local = {'model': 'local-model', 'reasoningEffort': None}
        self.status = self.resumed_status = 'idle'
        self.start_override = {}
        self.project_id = None
        self.project_supported = True
        self.errors = {}
        self.metadata_response = None
        self.project_pages = [{'data': [{'id': 'repo-project', 'name': 'Renamed project',
                               'roots': [{'path': '/repos/test/repo'}]}], 'nextCursor': None}]
        self.pages = [{'data': [
            {'model': 'gpt-6-astra', 'supportedReasoningEfforts': [
                {'reasoningEffort': value} for value in ('xhigh', 'max', 'ultra')]},
            {'model': 'gpt-5.6-terra', 'supportedReasoningEfforts': [{'reasoningEffort': 'medium'}]},
        ], 'nextCursor': None}]

    def send_notification(self, *args):
        pass

    def request(self, method, params):
        self.calls.append((method, params))
        if method in self.errors:
            raise self.errors[method]
        project_fields = {'projectId': self.project_id} if self.project_supported else {}
        if method == 'project/list':
            return self.project_pages[0 if params['cursor'] is None else int(params['cursor'])]
        if method == 'thread/metadata/update':
            if self.metadata_response is not None:
                return self.metadata_response
            self.project_id = params['projectId']
            return {'thread': {'id': params['threadId'], 'projectId': self.project_id}}
        if method == 'initialize':
            return {'codexHome': str(self.home), 'platformOs': 'linux'}
        if method == 'model/list':
            return self.pages[0 if params['cursor'] is None else int(params['cursor'])]
        if method == 'thread/read':
            return {'thread': {'status': self.status}}
        if method == 'thread/resume':
            return {'thread': {'id': 'saved', 'status': self.resumed_status, **project_fields}, **self.saved}
        if method == 'thread/start':
            return {'thread': {'id': 'new', 'status': 'idle', **project_fields},
                    'model': params.get('model', self.local['model']),
                    'reasoningEffort': params['config'].get('model_reasoning_effort', self.local['reasoningEffort']),
                    **self.start_override}
        if method == 'thread/name/set':
            return {}
        if method == 'turn/start':
            return {'turn': {'id': 'turn'}}
        raise AssertionError(f'Unexpected request: {method}')

    def drain_until_turn_complete(self):
        return {'id': 'turn', 'status': 'completed'}


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name)
        self.server, self.logs = Server(self.home), []

    def profile(self, name='custom', effort='ultra', model='gpt-6-astra', extra=''):
        path = self.home / f'{name}.config.toml'
        path.write_text(f'model = "{model}"\nmodel_reasoning_effort = "{effort}"\n{extra}')
        return path

    def launch(self, settings=None, *, role='execution', existing=False):
        thread_file = self.home / 'thread-id'
        if existing:
            thread_file.write_text('saved')
        (self.home / 'codex-settings.json').write_text(json.dumps(settings or {}))
        source = build_audit_client(WORKFLOW) if role == 'audit' else EXECUTOR
        tree = ast.parse(source)
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {'resolve_project_id', 'assign_thread_project'}]
        body = ast.Module(body=[*helpers, tree.body[-1]], type_ignores=[])
        env = {'CODEX_LAUNCH_ROLE': role, 'GITHUB_REPOSITORY': 'test/repo',
               'PR_NUMBER': '7', 'GITHUB_RUN_ATTEMPT': '1', 'REVIEW_HEAD_SHA': 'a' * 40,
               'SKILLFORGE_TASK_PROMPT': 'Review the exact target'}
        namespace = {
            'os': os, 'json': json, 'ProfilePolicy': ProfilePolicy, 'load_settings': load_settings,
            'traceback': SimpleNamespace(print_exc=lambda: None),
            'WebSocketUnix': lambda path: None, 'AppServerClient': lambda ws: self.server,
            'SOCKET_PATH': 'unused', 'WORKTREE': '/isolated/issue-worktree', 'ISSUE_NUMBER': '1',
            'ISSUE_TITLE': 'Test', 'RUN_ID': 'test', 'THREAD_FILE': str(thread_file),
            'STATE_DIR': str(self.home), 'REPO_ROOT': '/repos/test/repo',
            'log': lambda event, **fields: self.logs.append({'event': event, **fields}),
            'write_atomic': lambda path, value: Path(path).write_text(value),
            'status_is_active': lambda status: status == 'active' or status == {'type': 'active'},
        }
        for key in ('ERROR_FILE', 'TURN_FILE', 'READY_FILE', 'COMPLETED_FILE'):
            namespace[key] = str(self.home / key)
        with patch.dict(os.environ, env, clear=True):
            exec(compile(body, 'launcher-lifecycle', 'exec'), namespace)

    def methods(self):
        return [method for method, _ in self.server.calls]

    def params(self, method):
        return next(params for name, params in self.server.calls if name == method)

    def test_new_native_thread_needs_no_profiles_or_catalog(self):
        self.launch()
        start, turn = self.params('thread/start'), self.params('turn/start')
        self.assertEqual(start['config'], {CAP_KEY: 3})
        self.assertNotIn('model', start)
        self.assertNotIn('allowProviderModelFallback', start)
        self.assertNotIn('model', turn)
        self.assertNotIn('effort', turn)
        self.assertNotIn('model/list', self.methods())
        self.assertFalse(list(self.home.glob('*.config.toml')))
        self.assertEqual(turn['approvalPolicy'], 'on-request')
        self.assertEqual(turn['approvalsReviewer'], 'auto_review')
        self.assertEqual(turn['sandboxPolicy'], {'type': 'workspaceWrite',
                         'writableRoots': ['/isolated/issue-worktree'], 'networkAccess': True})

    def test_native_resume_does_not_reject_legacy_xhigh(self):
        self.launch(existing=True)
        self.assertNotIn('model', self.params('thread/resume'))
        self.assertNotIn('effort', self.params('turn/start'))
        self.assertNotIn('thread/start', self.methods())
        self.assertNotIn('model/list', self.methods())

    def test_native_resume_accepts_different_model_and_null_effort(self):
        self.server.saved = {'model': 'user-selected-provider/model', 'reasoningEffort': None}
        self.launch(existing=True)
        self.assertNotIn('model', self.params('turn/start'))
        record = next(item for item in self.logs if item['event'] == 'model_policy_resolved')
        self.assertEqual(record['mode'], 'native')
        self.assertIsNone(record['observedEffort'])
        self.assertNotIn('confirmedEffort', record)

    def test_direct_override_needs_no_profile_and_preserves_permissions(self):
        self.launch({'model': 'gpt-6-astra', 'model_reasoning_effort': 'ultra'})
        self.assertIs(self.params('thread/start')['allowProviderModelFallback'], False)
        self.assertEqual(self.params('turn/start')['model'], 'gpt-6-astra')
        self.assertEqual(self.params('turn/start')['effort'], 'ultra')
        self.assertEqual(self.params('thread/start')['sandbox'], 'workspace-write')

    def test_explicit_override_is_applied_to_inactive_old_selection(self):
        self.server.saved = {'model': 'different-model', 'reasoningEffort': 'low'}
        self.launch({'model': 'gpt-6-astra', 'model_reasoning_effort': 'ultra'}, existing=True)
        self.assertNotIn('model', self.params('thread/resume'))
        self.assertEqual(self.params('turn/start')['effort'], 'ultra')
        record = next(item for item in self.logs if item['event'] == 'model_policy_resolved')
        self.assertEqual(record['observedEffort'], 'low')
        self.assertEqual(record['turnOverrides']['effort'], 'ultra')
        self.assertNotIn('confirmedEffort', record)

    def test_explicit_lower_override_is_not_confused_with_silent_downgrade(self):
        self.server.saved['reasoningEffort'] = 'ultra'
        self.launch({'model_reasoning_effort': 'xhigh'}, existing=True)
        self.assertEqual(self.params('turn/start')['effort'], 'xhigh')
        self.assertNotIn('model', self.params('turn/start'))

    def test_profile_is_optional_and_inline_fields_win(self):
        self.profile(effort='xhigh', extra='[agents]\nmax_concurrent_threads_per_session = 3\n')
        self.launch({'profile': 'custom', 'model_reasoning_effort': 'ultra'})
        self.assertEqual(self.params('turn/start')['effort'], 'ultra')

    def test_profile_supports_other_available_models_without_astra_floor(self):
        self.profile(model='gpt-5.6-terra', effort='medium')
        self.launch({'profile': 'custom'})
        self.assertEqual(self.params('turn/start')['model'], 'gpt-5.6-terra')

    def test_model_only_override_preserves_unset_native_effort(self):
        self.launch({'model': 'gpt-6-astra'})
        self.assertNotIn('effort', self.params('turn/start'))

    def test_effort_only_override_uses_observed_model(self):
        self.server.local = {'model': 'gpt-6-astra', 'reasoningEffort': 'medium'}
        self.launch({'model_reasoning_effort': 'ultra'})
        self.assertNotIn('model', self.params('thread/start'))
        self.assertNotIn('model', self.params('turn/start'))

    def test_explicit_invalid_target_is_not_ignored(self):
        for settings in ({'model': 'missing'}, {'model': 'gpt-5.6-terra', 'model_reasoning_effort': 'ultra'}):
            with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, 'no fallback'):
                self.launch(settings)
        self.assertNotIn('turn/start', self.methods())

    def test_active_thread_is_never_resumed_or_steered(self):
        self.server.status = {'type': 'active'}
        with self.assertRaisesRegex(RuntimeError, 'already active'):
            self.launch({'model': 'gpt-6-astra', 'model_reasoning_effort': 'ultra'}, existing=True)
        self.assertNotIn('thread/resume', self.methods())
        self.assertNotIn('turn/start', self.methods())
        self.assertNotIn('thread/metadata/update', self.methods())

    def test_thread_becoming_active_after_resume_is_not_steered(self):
        self.server.resumed_status = 'active'
        with self.assertRaisesRegex(RuntimeError, 'became active'):
            self.launch(existing=True)
        self.assertNotIn('turn/start', self.methods())
        self.assertNotIn('thread/metadata/update', self.methods())

    def test_native_audit_is_fresh_without_requiring_profile(self):
        self.launch(role='audit')
        self.assertEqual(self.methods().count('thread/start'), 1)
        self.assertFalse(set(self.methods()) & {'thread/read', 'thread/resume', 'thread/fork'})
        self.assertNotIn('effort', self.params('turn/start'))
        self.assertEqual(self.params('turn/start')['input'][0]['text'], 'Review the exact target')

    def test_audit_can_use_explicit_issue_profile(self):
        self.profile()
        self.launch({'profile': 'custom'}, role='audit')
        self.assertEqual(self.params('turn/start')['effort'], 'ultra')

    def test_audit_refuses_existing_thread(self):
        with self.assertRaisesRegex(RuntimeError, 'never resume'):
            self.launch(role='audit', existing=True)
        self.assertFalse(set(self.methods()) & {'thread/start', 'thread/read', 'thread/resume', 'turn/start'})

    def test_new_thread_mismatch_does_not_start_turn(self):
        self.server.start_override = {'reasoningEffort': 'xhigh'}
        with self.assertRaisesRegex(ValueError, 'requested.*ultra.*observed.*xhigh'):
            self.launch({'model': 'gpt-6-astra', 'model_reasoning_effort': 'ultra'})
        self.assertNotIn('turn/start', self.methods())

    def test_profile_paths_missing_bad_toml_and_permissions_are_rejected(self):
        (self.home / 'malformed.config.toml').write_text('model = [')
        self.profile('permissions', extra='sandbox_mode = "danger-full-access"\n')
        self.profile('wrong-cap', extra='[agents]\nmax_concurrent_threads_per_session = 4\n')
        for name in ('../auth', 'missing', 'malformed', 'permissions', 'wrong-cap'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.launch({'profile': name})
        self.assertNotIn('thread/start', self.methods())

    def test_pagination_and_cycle_are_bounded(self):
        models = self.server.pages[0]
        self.server.pages = [{'data': [], 'nextCursor': '1'}, models]
        self.launch({'model': 'gpt-6-astra', 'model_reasoning_effort': 'ultra'})
        self.assertEqual([params['cursor'] for method, params in self.server.calls if method == 'model/list'], [None, '1'])
        self.server.calls.clear()
        self.server.pages = [{'data': [], 'nextCursor': '0'}]
        with self.assertRaisesRegex(ValueError, 'pagination'):
            self.launch({'model': 'gpt-6-astra'})
        self.assertNotIn('turn/start', self.methods())

    def test_native_mode_does_not_read_codex_home(self):
        policy = ProfilePolicy(self.server, None, {}, 'execution')
        policy.confirm({})
        self.assertEqual(policy.turn_options(), {})
        self.assertEqual(self.server.calls, [])

    def test_workflows_do_not_default_profiles_or_keep_actions_credentials(self):
        for workflow in (WORKFLOW, AUDIT):
            self.assertNotIn('CODEX_PROFILE_NAME', workflow)
            self.assertNotIn('vars.CODEX_', workflow)
            self.assertIn('prepare-issue "${state_dir}/codex-settings.json"', workflow)
            detached = workflow.split('RUNNER_TRACKING_ID="" nohup', 1)[1]
            self.assertIn('env -u GH_TOKEN -u GITHUB_TOKEN', detached)
            self.assertNotIn('GITHUB_TOKEN=', detached)
        self.assertIn('  issues: read', WORKFLOW)
        self.assertLess(WORKFLOW.index('if process_from_pid_file_is_alive'), WORKFLOW.index('prepare-issue'))


    def test_projects_cover_new_resumed_and_fresh_audit_threads(self):
        for role, existing in (('execution', False), ('execution', True), ('audit', False)):
            with self.subTest(role=role, existing=existing):
                self.setUp()
                self.launch(role=role, existing=existing)
                target = 'saved' if existing else 'new'
                self.assertEqual(self.params('thread/metadata/update'),
                                 {'threadId': target, 'projectId': 'repo-project'})
                methods = self.methods()
                self.assertLess(methods.index('thread/metadata/update'), methods.index('turn/start'))
                self.assertEqual(methods.count('thread/start'), 0 if existing else 1)
                self.assertEqual(methods.count('turn/start'), 1)
                self.assertEqual(self.params('turn/start')['environments'],
                                 [{'environmentId': 'local', 'cwd': '/isolated/issue-worktree'}])
                self.assertEqual(self.params('turn/start')['sandboxPolicy']['writableRoots'],
                                 ['/isolated/issue-worktree'])
                self.assertEqual(self.params('initialize')['capabilities'], {'experimentalApi': True})
                self.assertFalse(set(methods) & {'project/create', 'project/import', 'project/update'})
                if role == 'audit':
                    self.assertFalse(set(methods) & {'thread/read', 'thread/resume', 'thread/fork'})
                self.assertTrue(any(item['event'] == 'thread_project_assigned' for item in self.logs))

    def test_existing_project_assignments_are_never_retargeted(self):
        for project_id in ('manual-project', 'repo-project'):
            for role, existing in (('execution', True), ('audit', False)):
                with self.subTest(project=project_id, role=role):
                    self.setUp()
                    self.server.project_id = project_id
                    self.launch(role=role, existing=existing)
                    self.assertNotIn('thread/metadata/update', self.methods())
                    self.assertEqual(self.server.project_id, project_id)
                    self.assertTrue(any(item['event'] == 'thread_project_preserved' for item in self.logs))
                    self.assertIn('turn/start', self.methods())

    def test_manual_move_during_project_lookup_is_preserved_on_resume(self):
        request = self.server.request
        def move_then_request(method, params):
            if method == 'project/list':
                self.server.project_id = 'moved-in-desktop'
            return request(method, params)
        self.server.request = move_then_request
        self.launch(existing=True)
        self.assertNotIn('thread/metadata/update', self.methods())
        self.assertEqual(self.server.project_id, 'moved-in-desktop')

    def test_project_match_uses_all_pages_and_canonical_roots_not_names(self):
        link = self.home / 'repo-link'
        link.symlink_to('/repos/test/repo', target_is_directory=True)
        self.server.project_pages = [
            {'data': [], 'nextCursor': '1'},
            {'data': [{'id': 'correct', 'name': 'Anything',
                       'roots': [{'path': '/unrelated'}, {'path': str(link) + '/.'}]}],
             'nextCursor': None},
        ]
        self.launch()
        self.assertEqual(self.params('thread/metadata/update')['projectId'], 'correct')
        self.assertEqual([p['cursor'] for m, p in self.server.calls if m == 'project/list'], [None, '1'])

    def test_same_name_parent_prefix_and_issue_worktree_do_not_match(self):
        for path in ('/other/repo', '/repos/test', '/repos/test/repo-extra', '/isolated/issue-worktree'):
            with self.subTest(path=path):
                self.setUp()
                self.server.project_pages[0]['data'][0].update(
                    name='repo', roots=[{'path': path}])
                self.launch()
                self.assertNotIn('thread/metadata/update', self.methods())
                self.assertIn('turn/start', self.methods())
                self.assertTrue(any('found 0' in item.get('reason', '') for item in self.logs))

    def test_ambiguous_project_on_later_page_is_not_guessed(self):
        first = self.server.project_pages[0]
        first['nextCursor'] = '1'
        self.server.project_pages.append({'data': [{'id': 'duplicate',
            'roots': [{'path': '/repos/test/repo'}]}], 'nextCursor': None})
        self.launch(role='audit')
        self.assertNotIn('thread/metadata/update', self.methods())
        self.assertTrue(any('found 2' in item.get('reason', '') for item in self.logs))
        self.assertIn('turn/start', self.methods())

    def test_missing_project_field_does_not_mean_unassigned(self):
        self.server.project_supported = False
        self.launch(existing=True)
        self.assertNotIn('thread/metadata/update', self.methods())
        self.assertTrue(any('did not expose' in item.get('reason', '') for item in self.logs))
        self.assertIn('turn/start', self.methods())

    def test_unsupported_project_rpcs_warn_without_duplicate_launches(self):
        for method in ('project/list', 'thread/metadata/update'):
            for role in ('execution', 'audit'):
                with self.subTest(method=method, role=role):
                    self.setUp()
                    self.server.errors[method] = RuntimeError('method not found or experimental field unavailable')
                    self.launch(role=role)
                    self.assertEqual(self.methods().count('thread/start'), 1)
                    self.assertEqual(self.methods().count('turn/start'), 1)
                    self.assertFalse(any(item['event'] == 'thread_project_assigned' for item in self.logs))
                    self.assertTrue(any(item['event'] == 'thread_project_warning' and
                                        item['level'] == 'warning' for item in self.logs))

    def test_ignored_or_unconfirmed_assignment_is_not_reported_as_success(self):
        for response in ({}, {'thread': {'id': 'new', 'projectId': None}},
                         {'thread': {'id': 'other', 'projectId': 'repo-project'}},
                         {'thread': {'id': 'new', 'projectId': 'different'}}):
            with self.subTest(response=response):
                self.setUp()
                self.server.metadata_response = response
                self.launch()
                self.assertEqual(self.methods().count('thread/metadata/update'), 1)
                self.assertEqual(self.methods().count('thread/start'), 1)
                self.assertIn('turn/start', self.methods())
                self.assertFalse(any(item['event'] == 'thread_project_assigned' for item in self.logs))
                self.assertTrue(any('did not confirm' in item.get('reason', '') for item in self.logs))

    def test_invalid_project_pages_do_not_authorize_a_partial_match(self):
        cases = [None, [], {}, {'data': None, 'nextCursor': None},
                 {'data': [], 'nextCursor': 0}, {'data': [], 'nextCursor': ''},
                 {'data': [None], 'nextCursor': None},
                 {'data': [{'id': '', 'roots': []}], 'nextCursor': None},
                 {'data': [{'id': 'bad', 'roots': [{'path': 'relative'}]}], 'nextCursor': None},
                 {'data': [{'id': 'bad', 'roots': [None]}], 'nextCursor': None}]
        for page in cases:
            with self.subTest(page=page):
                self.setUp()
                self.server.project_pages = [page]
                self.launch()
                self.assertNotIn('thread/metadata/update', self.methods())
                self.assertIn('turn/start', self.methods())
                self.assertTrue(any(item['event'] == 'thread_project_warning' for item in self.logs))

    def test_project_pagination_cycle_and_page_limit_do_not_block_execution(self):
        for cycle in (True, False):
            with self.subTest(cycle=cycle):
                self.setUp()
                first = self.server.project_pages[0]
                first['nextCursor'] = '0' if cycle else '1'
                if not cycle:
                    self.server.project_pages.extend(
                        {'data': [], 'nextCursor': str(i + 1)} for i in range(1, 20))
                self.launch()
                self.assertNotIn('thread/metadata/update', self.methods())
                self.assertEqual(self.methods().count('project/list'), 2 if cycle else 20)
                self.assertIn('turn/start', self.methods())

    def test_transport_and_core_lifecycle_errors_are_not_swallowed(self):
        for method, error in (('project/list', OSError('disconnected')),
                              ('thread/metadata/update', EOFError('closed')),
                              ('thread/start', RuntimeError('start failed')),
                              ('turn/start', RuntimeError('turn failed'))):
            with self.subTest(method=method):
                self.setUp()
                self.server.errors[method] = error
                with self.assertRaises(type(error)):
                    self.launch()
                self.assertEqual(self.methods().count(method), 1)
                self.assertTrue(any(item['event'] == 'fatal' for item in self.logs))
                if method != 'turn/start':
                    self.assertNotIn('turn/start', self.methods())


class SettingsTests(unittest.TestCase):
    def test_absent_empty_and_null_body(self):
        for body in (None, '', 'Use Astra ultra in prose.', '```codex\n```', '```yaml\nmodel: example\n```'):
            self.assertEqual(issue_settings(body), {})

    def test_valid_dedicated_toml_fences(self):
        expected = {'model': 'gpt-6-astra', 'model_reasoning_effort': 'ultra'}
        for fence in ('```', '~~~~'):
            body = f'{fence}codex\nmodel = "gpt-6-astra"\nmodel_reasoning_effort = "ultra"\n{fence}\n'
            self.assertEqual(issue_settings(body), expected)

    def test_nested_quoted_and_indented_examples_are_not_settings(self):
        example = '```codex\nprofile = "not-selected"\n```\n'
        for body in ('````markdown\n' + example + '````',
                     '\n'.join('> ' + line for line in example.splitlines()),
                     textwrap.indent(example, '    ')):
            self.assertEqual(issue_settings(body), {})

    def test_duplicate_unclosed_unknown_and_nonstring_are_rejected(self):
        bad = ('```codex\n```\n```codex\n```', '```codex\nmodel = "a"',
               '```codex\nmodel = [\n```', '```codex\nsandbox_mode = "read-only"\n```',
               '```codex\nmodel = 42\n```', '```codex\nprofile = "../auth"\n```',
               '```codex\nmodel="a"\nmodel="b"\n```')
        for body in bad:
            with self.subTest(body=body), self.assertRaises(ValueError):
                issue_settings(body)

    def test_bounded_settings_and_json_snapshot(self):
        with self.assertRaisesRegex(ValueError, '8 KiB'):
            issue_settings('```codex\n#' + 'x' * 8200 + '\n```')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            path.write_text(json.dumps({'model': 'safe'}))
            self.assertEqual(load_settings(path), {'model': 'safe'})
            path.write_text(' ' * 8193)
            with self.assertRaisesRegex(ValueError, 'Oversized'):
                load_settings(path)

    def test_preparation_uses_get_and_writes_only_validated_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            doc = {'number': 114, 'body': 'Do not retain this prose.\n```codex\nmodel="gpt-6-astra"\n```'}
            response = io.BytesIO(json.dumps(doc).encode())
            env = {'GITHUB_REPOSITORY': 'test/repo', 'ISSUE_NUMBER': '114', 'GITHUB_TOKEN': 'test-secret'}
            with patch.dict(os.environ, env, clear=True), patch('urllib.request.urlopen', return_value=response) as get:
                prepare_issue(path)
            request = get.call_args.args[0]
            self.assertEqual(request.get_method(), 'GET')
            self.assertEqual(request.full_url, 'https://api.github.com/repos/test/repo/issues/114')
            self.assertEqual(get.call_args.kwargs['timeout'], 30)
            self.assertEqual(load_settings(path), {'model': 'gpt-6-astra'})
            self.assertNotIn('test-secret', path.read_text())
            self.assertNotIn('prose', path.read_text())
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_failed_issue_read_does_not_fall_back_or_replace_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'settings.json'
            path.write_text('previous')
            env = {'GITHUB_REPOSITORY': 'test/repo', 'ISSUE_NUMBER': '114', 'GITHUB_TOKEN': 'test-secret'}
            with patch.dict(os.environ, env, clear=True), patch('urllib.request.urlopen', side_effect=OSError('offline')):
                with self.assertRaises(OSError):
                    prepare_issue(path)
            self.assertEqual(path.read_text(), 'previous')


if __name__ == '__main__':
    unittest.main()
