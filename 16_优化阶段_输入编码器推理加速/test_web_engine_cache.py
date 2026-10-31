import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from web_engine_cache import build_cached


class EngineCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='glc_engine_test_');self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))

    def tearDown(self):self.temp.cleanup()

    def factory(self,model,x,variant,path):
        path.mkdir(parents=True);(path/'trt_fp16.engine').write_bytes(b'complete engine')
        return SimpleNamespace(setup_seconds=0.)

    def test_publishes_only_completed_engine(self):
        build_cached(None,None,'trt_fp16_stream',self.root/'cache',self.root/'stage',lambda:None,self.factory)
        self.assertEqual((self.root/'cache/trt_fp16.engine').read_bytes(),b'complete engine')

    def test_cancel_before_publication_leaves_no_cache(self):
        def stop():raise InterruptedError()
        with self.assertRaises(InterruptedError):build_cached(None,None,'trt_fp16_stream',self.root/'cache',self.root/'stage',stop,self.factory)
        self.assertFalse((self.root/'cache/trt_fp16.engine').exists())

    def test_cache_hit_does_not_rebuild_in_staging(self):
        cache=self.root/'cache';cache.mkdir();(cache/'trt_fp16.engine').write_bytes(b'old valid engine')
        calls=[]
        def load(m,x,v,path):calls.append(path);return SimpleNamespace(setup_seconds=0.)
        build_cached(None,None,'trt_fp16_stream',cache,self.root/'stage',lambda:None,load)
        self.assertEqual(calls,[cache]);self.assertEqual((cache/'trt_fp16.engine').read_bytes(),b'old valid engine')


if __name__=='__main__':unittest.main()
