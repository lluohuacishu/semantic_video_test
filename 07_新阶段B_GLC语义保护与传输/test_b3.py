"""Budget selection and common-region evaluation checks without GPU runs."""
import unittest
import numpy as np
from encode_b3_controls import next_integer, closest
from evaluate_b3_controls import rect_mask, edges


class BudgetTests(unittest.TestCase):
    def test_increasing_and_decreasing_brackets(self):
        self.assertEqual(next_integer([{"q":0,"b":100},{"q":63,"b":900}],500,"q","b",0,63),31)
        self.assertEqual(next_integer([{"q":0,"b":900},{"q":128,"b":100}],500,"q","b",0,204),64)

    def test_no_bracket_does_not_extrapolate(self):
        self.assertIsNone(next_integer([{"q":0,"b":100},{"q":63,"b":200}],500,"q","b",0,63))
        self.assertIsNone(next_integer([{"q":30,"b":510},{"q":31,"b":490}],500,"q","b",0,63))

    def test_selection_ignores_quality_and_tie_prefers_smaller(self):
        rows=[{"b":510,"psnr":100},{"b":490,"psnr":0}]
        self.assertEqual(closest(rows,500,"b"),rows[1])

    def test_geometry_area_and_window_end(self):
        patches=[{"start":224,"frames":26,"box":[0,180,640,360]}, {"start":224,"frames":26,"box":[640,180,640,360]}]
        self.assertEqual(int(rect_mask(patches,249).sum()),1280*720//2)
        self.assertFalse(rect_mask(patches,250).any())
        self.assertFalse(rect_mask(patches,223).any())

    def test_edge_diagnostic_detects_patch_step(self):
        patches=[{"start":0,"frames":1,"box":[2,2,4,4]}]
        error=np.zeros((8,8,3),dtype=np.float32)
        self.assertEqual(edges(error,patches,0),0)
        error[2:6,2:6]=10
        self.assertEqual(edges(error,patches,0),10)


if __name__ == "__main__":
    unittest.main()
