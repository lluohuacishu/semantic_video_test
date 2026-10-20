"""C3 HTTP, evidence mapping and job lifecycle regression. No neural inference."""
import copy
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from c3_catalog import Catalog, transmission
from c3_server import C3App, C3Store, DirectoryLease, Server, parse_options, verify_result, write


class C3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = Catalog()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = C3App(self.temp.name, self.catalog)
        self.server = Server(('127.0.0.1', 0), self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.app.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def call(self, method, route, value=None, headers=None, raw=None):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        body = json.dumps(value).encode() if value is not None else raw
        h = {'Content-Type': 'application/json'} if value is not None else {}
        h.update(headers or {})
        connection.request(method, route, body, h)
        response = connection.getresponse()
        result = response.status, dict(response.getheaders()), response.read()
        connection.close()
        return result

    def reserve(self):
        return self.app.reserve({'source_asset': 'source.main', 'options': {'preset': 'compact', 'frames': 8}})

    def test_all_four_and_second_project_from_evidence(self):
        self.assertEqual(5, len(self.catalog.records))
        self.assertEqual({1, 3}, {r['performance']['complete_pair_count'] for r in self.catalog.records})
        self.assertTrue(all(r['checks']['stream_matches_reference'] for r in self.catalog.records))
        self.assertFalse(self.catalog.payload['speed_goals']['challenge_2x_met'])

    def test_theoretical_transport_matches_wire_header_sizes(self):
        for size in (9999, 10000, 26397, 532330):
            result = transmission(size, 'a' * 64)
            self.assertEqual(size + 4 + len(json.dumps({'bytes':size,'sha256':'a'*64},separators=(',',':')).encode()),result['application_bytes'])
            self.assertAlmostEqual(result['application_bytes'] * 8 / 20000,result['theoretical_seconds']['20'])

    def test_no_automatic_preset(self):
        for options in ({}, {'preset':'bad'}, {'preset':'compact','q':63}, {'preset':'compact','frames':True}, {'preset':'compact','frames':126}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                parse_options(options,self.catalog.presets['presets'])

    def test_python_fallback_is_explicit(self):
        self.assertEqual('python',parse_options({'preset':'motion','backend':'python'},self.catalog.presets['presets'])['backend'])

    def test_open_catalog_never_submits_work(self):
        with patch.object(self.app.executor,'submit') as submit:
            self.assertEqual(200,self.call('GET','/api/config')[0])
            self.assertEqual(200,self.call('GET','/api/catalog')[0])
            self.assertEqual([],self.app.store.list())
            submit.assert_not_called()

    def test_stage_does_not_run_and_cancel_is_terminal(self):
        with patch.object(self.app.executor,'submit') as submit:
            job=self.reserve()
            self.assertEqual('staged',job['status'])
            self.assertEqual('cancelled',self.app.cancel(job['id'])['status'])
            submit.assert_not_called()
            with self.assertRaises(ValueError): self.app.start(job['id'])

    def test_start_cannot_be_submitted_twice(self):
        job=self.reserve()
        with patch.object(self.app.executor,'submit') as submit:
            self.assertEqual('queued',self.app.start(job['id'])['status'])
            with self.assertRaises(ValueError): self.app.start(job['id'])
            submit.assert_called_once()

    def test_upload_is_separate_from_start(self):
        status,_,raw=self.call('POST','/api/jobs/reserve',{'filename':'short.mp4','bytes':4,'options':{'preset':'detail'}})
        self.assertEqual(201,status)
        job=json.loads(raw)
        with patch.object(self.app.executor,'submit') as submit:
            status,_,raw=self.call('POST',f"/api/jobs/{job['id']}/source",raw=b'abcd')
            self.assertEqual(200,status)
            self.assertEqual('staged',json.loads(raw)['status'])
            submit.assert_not_called()

    def test_cancelled_upload_cannot_be_completed(self):
        job=self.app.reserve({'filename':'test.mp4','bytes':4,'options':{'preset':'compact'}})
        self.app.cancel(job['id'])
        self.assertEqual(400,self.call('POST',f"/api/jobs/{job['id']}/source",raw=b'abcd')[0])

    def test_wrong_upload_length_rejected(self):
        job=self.app.reserve({'filename':'test.mp4','bytes':4,'options':{'preset':'compact'}})
        self.assertEqual(400,self.call('POST',f"/api/jobs/{job['id']}/source",raw=b'ab')[0])

    def test_unknown_source_and_upload_type_rejected(self):
        for value in ({'source_asset':'C:/secret','options':{'preset':'compact'}},
                      {'filename':'payload.py','bytes':4,'options':{'preset':'compact'}}):
            self.assertEqual(400,self.call('POST','/api/jobs/reserve',value)[0])

    def test_range_and_head_support_video_seek(self):
        route='/assets/c2.compact.independent_decoded_preview'
        status,headers,data=self.call('GET',route,headers={'Range':'bytes=0-31'})
        self.assertEqual(206,status);self.assertEqual(32,len(data));self.assertTrue(headers['Content-Range'].startswith('bytes 0-31/'))
        self.assertEqual(416,self.call('GET',route,headers={'Range':'bytes=999999999-'})[0])
        self.assertEqual(b'',self.call('HEAD',route)[2])

    def test_report_download_handles_chinese_filename(self):
        status,headers,data=self.call('GET','/assets/report.c2?download=1')
        self.assertEqual(200,status);self.assertIn("filename*=UTF-8''",headers['Content-Disposition'])
        self.assertIn('C2',data.decode('utf-8'))

    def test_origin_and_path_allowlist(self):
        self.assertEqual(403,self.call('GET','/api/config',headers={'Origin':'https://external.invalid'})[0])
        self.assertEqual(403,self.call('POST','/api/jobs/reserve',{},headers={'Origin':'https://external.invalid'})[0])
        self.assertEqual(404,self.call('GET','/assets/../../交接文档.md'.replace('交接文档','secret'))[0])

    def test_duplicate_store_service_is_rejected(self):
        with self.assertRaises(RuntimeError): DirectoryLease(self.temp.name)

    def test_missing_runtime_does_not_break_replay(self):
        self.app.runtime['python']=Path(self.temp.name)/'missing.exe'
        self.assertFalse(self.app.config()['runtime_available'])
        self.assertEqual(200,self.call('GET','/api/catalog')[0])
        with self.assertRaises(ValueError): self.app.start(self.reserve()['id'])

    def test_python_fallback_without_native_dll(self):
        self.app.runtime['native_library']=Path(self.temp.name)/'missing.dll'
        self.assertTrue(self.app.config()['runtime_available'])
        self.assertFalse(self.app.config()['native_available'])
        with self.assertRaises(ValueError): self.app.start(self.reserve()['id'])
        job=self.app.reserve({'source_asset':'source.main','options':{'preset':'compact','backend':'python'}})
        with patch.object(self.app.executor,'submit'):
            self.assertEqual('queued',self.app.start(job['id'])['status'])

    def test_missing_media_is_explicitly_unavailable(self):
        key='history.b2.comparison'
        with patch.dict(self.catalog.assets,{key:{**self.catalog.assets[key],'available':False}}):
            self.assertFalse(self.catalog.asset_ref(key)['available'])
            self.assertEqual(404,self.call('GET','/assets/'+key)[0])

    def test_error_and_cancel_release_serial_queue(self):
        active=threading.Event()
        def execute(role,task,folder):
            active.set()
            if not self.app.store.events[folder.name].wait(4): raise RuntimeError('fixture failure')
            raise InterruptedError('fixture cancellation')
        with patch.object(self.app,'execute',side_effect=execute):
            first=self.reserve();self.app.start(first['id']);self.assertTrue(active.wait(3))
            second=self.reserve();self.app.start(second['id'])
            self.assertEqual('queued',self.app.store.get(second['id'])['status'])
            self.app.cancel(first['id']);self.app.cancel(second['id'])
            self.app.executor.submit(lambda:None).result(timeout=5)
            self.assertEqual('cancelled',self.app.store.get(first['id'])['status'])
        with patch.object(self.app,'execute',side_effect=RuntimeError('fixture failure')):
            third=self.reserve();self.app.start(third['id']);self.app.executor.submit(lambda:None).result(timeout=5)
            self.assertEqual('error',self.app.store.get(third['id'])['status'])

    def test_refresh_returns_existing_job_without_rerun(self):
        job=self.reserve()
        with patch.object(self.app.executor,'submit') as submit:
            self.assertEqual('staged',json.loads(self.call('GET','/api/jobs/'+job['id'])[2])['status'])
            submit.assert_not_called()

    def test_restart_marks_unfinished_job_interrupted(self):
        job=self.reserve()
        self.app.store.update(job['id'],status='running')
        recovered=C3Store(self.temp.name)
        self.assertEqual('interrupted',recovered.get(job['id'])['status'])
        self.assertFalse(recovered.events)

    def test_uncertain_process_blocks_new_work(self):
        self.app.blockers.append('unresolved-worker')
        with self.assertRaises(ValueError): self.app.start(self.reserve()['id'])
        self.assertEqual(200,self.call('GET','/api/catalog')[0])

    def test_supervisor_creates_fresh_process_folder(self):
        job=self.reserve();folder=Path(self.temp.name)/job['id']
        def supervised(argv,process_folder,cwd,stop_file,poll_seconds):
            self.assertFalse(process_folder.exists())
            config=json.loads(Path(argv[-1]).read_text('utf-8'))
            self.assertEqual(str(process_folder),config['process_folder'])
            self.assertFalse(Path(argv[-1]).is_relative_to(process_folder))
            process_folder.mkdir()
            return {'returncode':0}
        with patch('c3_server.supervise',side_effect=supervised),patch('c3_server.inspect',return_value={'observed_state':'completed'}):
            self.app.execute('encode',{'id':'encode'},folder)
        self.assertFalse(self.app.blockers)

    def test_incomplete_outputs_not_downloadable(self):
        job=self.reserve();folder=Path(self.temp.name)/job['id'];(folder/'received.glcv').write_bytes(b'partial')
        self.assertEqual(404,self.call('GET',f"/jobs/{job['id']}/stream.glcv")[0])

    def test_browser_measurement_validates_status_and_numbers(self):
        job=self.reserve()
        self.assertEqual(400,self.call('POST',f"/api/jobs/{job['id']}/browser",{'preview_load_seconds':1})[0])
        self.app.store.update(job['id'],status='done')
        self.assertEqual(400,self.call('POST',f"/api/jobs/{job['id']}/browser",{'preview_load_seconds':float('nan')})[0])
        self.assertEqual(200,self.call('POST',f"/api/jobs/{job['id']}/browser",{'preview_load_seconds':1,'dispatch_to_preview_seconds':None,'after_refresh':True})[0])

    def test_verifier_rejects_frame_drift(self):
        record=self.catalog.records[-1]
        prefix='c2.second.compact.'
        enc=self.catalog.json_asset(prefix+'encode_result');dec=self.catalog.json_asset(prefix+'decode_result')
        folder=Path(self.temp.name)/'verify';(folder/'encode/encoder').mkdir(parents=True);(folder/'decode/receiver').mkdir(parents=True)
        (folder/'received.glcv').write_bytes(self.catalog.path(prefix+'stream').read_bytes())
        write(folder/'encode/encoder/encode.json',enc);write(folder/'decode/receiver/decode.json',dec)
        verify_result(folder)
        dec['frames'][0]['rgb_sha256']='bad';write(folder/'decode/receiver/decode.json',dec)
        with self.assertRaises(ValueError): verify_result(folder)


if __name__ == '__main__':
    unittest.main()
