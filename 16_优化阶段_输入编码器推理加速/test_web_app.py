"""No-GPU checks for mode admission, uploads, shared queue and result visibility."""
import io
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from web_app import InferenceLab,options
from experiment_common import load_settings,file_sha,np


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='glc_web_test_');self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        self.owner=SimpleNamespace(directory=self.root,role='sender',closed=False,blockers=[],
            runtime=load_settings()['runtime'],executor=SimpleNamespace(submit=Mock()))
        self.lab=InferenceLab(self.owner)
        self.value=dict(sample='long108_q0_640',modes=['eager','trt_fp16_stream'],frames=8,size=640,q=0)

    def tearDown(self):
        self.lab.close();self.assertTrue(self.root.name.startswith('glc_web_test_'));self.temp.cleanup()

    def test_modes_and_bounds(self):
        for modes in [[],['bad'],['eager','eager']]:
            with self.assertRaises(ValueError):options(dict(self.value,modes=modes))
        for change in [dict(frames=0),dict(frames=10001),dict(q=64),dict(q=True),dict(size=1280)]:
            with self.assertRaises(ValueError):options(dict(self.value,**change))

    def test_open_does_not_submit_gpu(self):
        self.assertTrue(self.lab.config()['ready']);self.assertFalse(self.lab.config()['gpu_started_on_open'])
        self.owner.executor.submit.assert_not_called()

    def test_full_video_has_no_frame_cutoff(self):
        result=options(dict(self.value,range='full'))
        self.assertEqual(result['range'],'full');self.assertIsNone(result['frames'])

    def test_frame_rate_modes_and_legacy_default(self):
        self.assertEqual(options(self.value)['fps_mode'],'original')
        result=options(dict(self.value,fps_mode='half',range='full'))
        self.assertEqual(result['fps_mode'],'half');self.assertIsNone(result['frames'])
        for value in ['fast',2,False,None,[]]:
            with self.assertRaises(ValueError):options(dict(self.value,fps_mode=value))

    def test_decoder_modes_and_legacy_default(self):
        self.assertEqual(options(self.value)['decoder_modes'],['eager'])
        expected=['eager','trt_fp32','trt_fp16_safe']
        self.assertEqual(options(dict(self.value,decoder_modes=expected))['decoder_modes'],expected)
        for value in [[],None,'eager',['bad'],['eager','eager'],[[]]]:
            with self.assertRaises(ValueError):options(dict(self.value,decoder_modes=value))

    def test_three_decoders_reuse_one_encoded_stream_and_keep_unique_videos(self):
        path=self.root/'source.npy';np.save(path,np.zeros((2,2,2,3),dtype=np.uint8))
        storage=dict(source=str(path),source_sha256=file_sha(path),frames=2,width=2,height=2,fps=15,
                     source_fps=30,frame_stride=2,duration_seconds=2/15,read_until_eof=True)
        common=dict(frames=2,core_seconds=1.,fps=2.,model_load_seconds=.1,engine_setup_seconds=.2,
                    warmup_seconds=.1,worker_seconds=2.,stream_sha256='same',stream_bytes=100,
                    reference_checksums={'latent_sha256':'same'},stream_file='stream.glcsnd',segments=1)
        calls=[]
        def execute(identifier,name,cfg):
            calls.append((name,dict(cfg)))
            if cfg['action']=='prepare':return dict(storage)
            if cfg['action']=='sender':return dict(common)
            return dict(common,reference_verified=True,decoded_storage=storage)
        value=dict(self.value,modes=['eager'],decoder_modes=['eager','trt_fp32','trt_fp16_safe'])
        j=self.lab.create(value)
        with patch.object(self.lab,'execute',side_effect=execute):self.lab.run(j['id'])
        result=self.lab.get(j['id']);self.assertEqual(result['status'],'done',result.get('error'))
        self.assertEqual(len([c for _,c in calls if c['action']=='sender']),1)
        receivers=[c for _,c in calls if c['action']=='receiver']
        self.assertEqual(len(receivers),3);self.assertEqual(len({c['input'] for c in receivers}),1)
        self.assertEqual(len({r['video'] for r in result['results']}),3)
        self.assertEqual(len({r['stream'] for r in result['results']}),1)
        for row in result['results']:
            name=row['video'].rsplit('/',1)[-1]
            self.assertEqual(self.lab.asset(j['id'],name),self.lab.directory/j['id']/row['receiver_folder']/'video.mp4')

    def test_retired_process_blocker_is_rechecked_without_gpu(self):
        self.owner.blockers=['retired-process']
        with patch('web_app.inspect',return_value={'observed_state':'cancelled'}):
            self.assertTrue(self.lab.config()['ready']);self.assertEqual(self.owner.blockers,[])
            self.assertEqual(self.lab.create(self.value)['status'],'staged')
        self.owner.executor.submit.assert_not_called()

    def test_active_or_unknown_process_blocker_is_preserved(self):
        for state in ['running','orphan_running','liveness_unknown']:
            self.owner.blockers=['guarded-process']
            with patch('web_app.inspect',return_value={'observed_state':state}):
                self.assertFalse(self.lab.config()['ready'])
                with self.assertRaises(ValueError):self.lab.create(self.value)
            self.assertEqual(self.owner.blockers,['guarded-process'])
        with patch('web_app.inspect',side_effect=OSError('identity unavailable')):
            self.assertFalse(self.lab.config()['ready'])
        self.owner.executor.submit.assert_not_called()

    def test_reuses_owner_queue_and_rejects_double_start(self):
        j=self.lab.create(self.value);self.lab.start(j['id'])
        self.owner.executor.submit.assert_called_once_with(self.lab.run,j['id'])
        with self.assertRaises(ValueError):self.lab.start(j['id'])

    def test_cancel_queued_never_starts_worker(self):
        j=self.lab.create(self.value);self.lab.start(j['id']);self.lab.cancel(j['id']);self.lab.run(j['id'])
        self.assertEqual(self.lab.get(j['id'])['status'],'cancelled')
        self.assertFalse((self.lab.directory/j['id']/'prepare').exists())

    def test_result_gate_and_traversal(self):
        j=self.lab.create(self.value)
        for name in ['eager.mp4','../../private','source.mp4']:
            with self.assertRaises(KeyError):self.lab.asset(j['id'],name)

    def test_upload_is_required_and_length_checked(self):
        value={k:v for k,v in self.value.items() if k!='sample'};value.update(filename='test.mp4',bytes=3)
        j=self.lab.create(value)
        with self.assertRaises(ValueError):self.lab.start(j['id'])
        h=SimpleNamespace(headers={'Content-Length':'2'},rfile=io.BytesIO(b'abc'),connection=SimpleNamespace(settimeout=lambda _:None))
        with self.assertRaises(ValueError):self.lab.upload(h,j['id'])
        h.headers={'Content-Length':'3'};self.assertEqual(self.lab.upload(h,j['id'])['status'],'staged')
        self.assertEqual((self.lab.directory/j['id']/'upload.mp4').read_bytes(),b'abc')
        reused=self.lab.create(dict(self.value,sample=None,source_job=j['id'],range='full'))
        self.assertEqual(reused['status'],'staged');self.assertEqual(reused['source_job'],j['id'])


if __name__=='__main__':unittest.main()
