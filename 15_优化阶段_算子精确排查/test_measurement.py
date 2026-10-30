"""Measurement accounting/ABI tests; no neural inference or GPU required."""
import ctypes as C
import unittest

from cupti_trace import Kernel9,Memcpy5,Memset4,API,External,Marker2,CudaEvent,union_ns
from nvtx_binding import pair_ranges,inputs_from_name


class MeasurementTests(unittest.TestCase):
    def test_api22_record_layouts(self):
        expected={Kernel9:208,Memcpy5:88,Memset4:88,API:40,External:24,Marker2:56,CudaEvent:24}
        for kind,size in expected.items():self.assertEqual(C.sizeof(kind),size)
        self.assertEqual(Kernel9.start.offset,16)
        self.assertEqual(Kernel9.name.offset,104)
        self.assertEqual(Memcpy5.start.offset,16)

    def test_interval_union_excludes_double_counting(self):
        self.assertEqual(union_ns([(0,10),(5,15),(30,40),(15,20)]),30)
        self.assertEqual(union_ns([]),0)
        self.assertEqual(union_ns([(1,1)]),0)
        with self.assertRaises(ValueError):union_ns([(2,1)])

    def test_native_ranges_and_shapes(self):
        rows=[dict(kind=12,flags=2,id=1,name='aten::mm, sizes = [[2, 3], [3, 4]], input_op_ids = []',timestamp=10,objectId=[3,4,0]),
              dict(kind=12,flags=4,id=1,name='',timestamp=20,objectId=[3,4,0])]
        ranges=pair_ranges(rows)
        self.assertEqual(ranges[0]['thread_id'],4)
        self.assertEqual(inputs_from_name(ranges[0]['name']),[[2,3],[3,4]])
        self.assertEqual(inputs_from_name('glc.script:swish|sizes=[[1,2]]'),[[1,2]])

    def test_negative_native_range_is_rejected(self):
        rows=[dict(kind=12,flags=2,id=1,name='a',timestamp=20,objectId=[1,2,0]),
              dict(kind=12,flags=4,id=1,name='',timestamp=10,objectId=[1,2,0])]
        with self.assertRaises(ValueError):pair_ranges(rows)


if __name__=='__main__':unittest.main()
