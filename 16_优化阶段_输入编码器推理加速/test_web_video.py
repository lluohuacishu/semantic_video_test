import tempfile
from pathlib import Path
import unittest
import numpy as np
from web_frames import FrameWriter,Frames,discard_working_frames,even_rgb
from web_transport import segment_ranges,package,unpack


class FullVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='glc_full_test_');self.root=Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(Path(tempfile.gettempdir()).resolve()))

    def tearDown(self):self.temp.cleanup()

    def test_partial_last_chunk_is_not_truncated_or_padded_as_video(self):
        w=FrameWriter(self.root/'chunks',4,3)
        for i in range(130):w.add(np.full((3,4,3),i,dtype='uint8'))
        meta=w.finish();self.assertEqual([c['frames'] for c in meta['chunks']],[64,64,2]);f=Frames(meta)
        for i in [0,63,64,127,128,129]:self.assertTrue(np.all(f[i]==i))
        with self.assertRaises(IndexError):f[130]
        f.close();discard_working_frames(meta,self.root);self.assertFalse(Path(meta['chunks'][0]['path']).exists())

    def test_cleanup_cannot_remove_external_frames(self):
        foreign=self.root/'source.npy';foreign.write_bytes(b'preserve')
        with self.assertRaises(ValueError):discard_working_frames({'chunks':[{'path':str(foreign)}]},self.root/'job')
        self.assertTrue(foreign.exists())

    def test_odd_dimensions_preserve_pixels_and_even_frames_are_unchanged(self):
        a=np.arange(3*5*3,dtype='uint8').reshape(3,5,3);b=even_rgb(a)
        self.assertEqual(b.shape,(4,6,3));self.assertTrue(np.array_equal(b[:3,:5],a))
        self.assertTrue(np.array_equal(b[-1,:5],a[-1]));self.assertTrue(np.array_equal(b[:3,-1],a[:,-1]))
        original=np.zeros((360,640,3),dtype='uint8');self.assertIs(even_rgb(original),original)

    def test_long_ranges_cover_every_frame_at_natural_gops(self):
        ranges=segment_ranges(10005)
        self.assertEqual(ranges,[(0,4096),(4096,8192),(8192,10005)])
        self.assertEqual(sum(b-a for a,b in ranges),10005)
        self.assertTrue(all(a%64==0 for a,b in ranges))

    def test_bundle_preserves_all_segments(self):
        segments=[]
        for i,n in enumerate([64,2]):
            f=self.root/f'segment_{i:06d}.glcsnd';f.write_bytes(bytes([i+1])*200)
            segments.append(dict(path=str(f),frames=n))
        path=package(self.root,segments,dict(frames=66,width=640,height=360,fps=12.5,q=0))
        files,meta=unpack(path,self.root/'received')
        self.assertEqual(meta['frames'],66)
        self.assertEqual([f.read_bytes() for f in files],[Path(s['path']).read_bytes() for s in segments])


if __name__=='__main__':unittest.main()
