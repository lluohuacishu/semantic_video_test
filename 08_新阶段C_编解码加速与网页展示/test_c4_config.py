"""Portable settings, upload-only operation, and durable transition checks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from c4_config import load_settings, UploadCatalog, discover_workspace
from c3_server import C3App, C3Store


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='glc C4 ')
        self.base = Path(self.temp.name)
        self.config = self.base / 'settings.json'

    def tearDown(self):
        self.temp.cleanup()

    def settings(self, value, env=None):
        self.config.write_text(json.dumps(value), encoding='utf-8-sig')
        return load_settings(self.config, environ=env or {})

    def test_relative_paths_resolve_against_config_not_cwd(self):
        s = self.settings({'workspace':'workspace', 'data_dir':'任务', 'runtime':{'python':'venv/python.exe'}})
        self.assertEqual(self.base/'workspace', s['workspace'])
        self.assertEqual(self.base/'任务', s['data_dir'])
        self.assertEqual(self.base/'venv/python.exe', s['runtime']['python'])

    def test_environment_overrides_config(self):
        s = self.settings({'data_dir':'old', 'runtime':{'python':'old.exe'}},
                          {'GLC_DATA_DIR':'new', 'GLC_PYTHON':'new.exe'})
        self.assertEqual(self.base/'new', s['data_dir'])
        self.assertEqual(self.base/'new.exe', s['runtime']['python'])

    def test_unknown_keys_fail_early(self):
        for value in ({'bad':True}, {'runtime':{'weights_typo':'x'}}, {'history_enabled':'false'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.settings(value)

    def test_empty_paths_rejected(self):
        with self.assertRaises(ValueError):
            self.settings({'runtime':{'python':''}})

    def test_missing_explicit_configuration_rejected(self):
        with self.assertRaises(ValueError):
            load_settings(self.base/'missing.json', environ={})

    def test_explicit_cli_config_takes_precedence_over_bad_environment_config(self):
        self.settings({'history_enabled':False})
        script=Path(__file__).with_name('c3_server.py')
        result=subprocess.run([sys.executable,'-B',str(script),'--config',str(self.config),'--check'],
                              env=dict(os.environ,GLC_CONFIG=str(self.base/'missing.json'),PYTHONIOENCODING='utf-8'),
                              capture_output=True,text=True,encoding='utf-8')
        self.assertEqual(0,result.returncode,result.stderr)
        self.assertEqual(str(self.config),json.loads(result.stdout)['config'])

    def test_bom_and_absolute_paths_supported(self):
        s=self.settings({'data_dir':str(self.base/'absolute')})
        self.assertEqual(self.base/'absolute', s['data_dir'])

    def test_shallow_checkout_does_not_require_original_directory_depth(self):
        self.assertEqual(Path('C:/demo'), discover_workspace(Path('C:/demo')))

    def test_workspace_discovery_uses_named_layout(self):
        (self.base/'01_项目/运行环境').mkdir(parents=True)
        repo=self.base/'01_项目/代码/moved'
        self.assertEqual(self.base,discover_workspace(repo))

    def test_upload_catalog_needs_no_archive(self):
        catalog=UploadCatalog()
        self.assertEqual(4,len(catalog.presets['presets']))
        self.assertFalse(catalog.payload['history_available'])
        with self.assertRaises(FileNotFoundError): catalog.path('source.main')

    def test_upload_only_app_stages_without_inference(self):
        app=C3App(self.base/'jobs', catalog=UploadCatalog())
        try:
            with patch.object(app.executor,'submit') as submit:
                job=app.reserve({'filename':'ordinary.mp4','bytes':4,'options':{'preset':'compact'}})
                app.source(job).write_bytes(b'test')
                app.store.update(job['id'],status='staged')
                self.assertFalse(app.config()['history_available'])
                self.assertEqual('staged',app.store.get(job['id'])['status'])
                submit.assert_not_called()
        finally: app.close()

    def test_transition_history_records_initial_queue_cancel(self):
        app=C3App(self.base/'jobs', catalog=UploadCatalog())
        try:
            job=app.reserve({'filename':'ordinary.mp4','bytes':4,'options':{'preset':'compact'}})
            app.store.update(job['id'],status='staged')
            app.store.update(job['id'],status='queued')
            app.cancel(job['id'])
            rows=[json.loads(line) for line in (app.directory/job['id']/'events.jsonl').read_text('utf-8').splitlines()]
            self.assertEqual(['uploading','staged','queued','cancelled'], [r['status'] for r in rows])
            self.assertTrue(app.store.get(job['id'])['cancel_requested_at'])
        finally: app.close()

    def test_recovery_keeps_completed_job_and_transition_evidence(self):
        app=C3App(self.base/'jobs', catalog=UploadCatalog())
        job=app.reserve({'filename':'ordinary.mp4','bytes':4,'options':{'preset':'compact'}})
        app.cancel(job['id'])
        before=(app.directory/job['id']/'events.jsonl').read_bytes()
        app.close()
        recovered=C3Store(self.base/'jobs')
        self.assertEqual('cancelled',recovered.get(job['id'])['status'])
        self.assertEqual(before,(self.base/'jobs'/job['id']/'events.jsonl').read_bytes())


if __name__ == '__main__':
    unittest.main()
