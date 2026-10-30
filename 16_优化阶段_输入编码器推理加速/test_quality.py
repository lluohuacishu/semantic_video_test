"""Quality aggregation and ROI gates must not exaggerate reconstruction quality."""
import math
import unittest
import numpy as np
from evaluate_experiment import quality


class QualityTests(unittest.TestCase):
    def test_aggregate_mse_before_db(self):
        g=np.full((3,2,2,3),32,dtype='uint8');b=g-1;a=g.copy();a[0]-=1;a[1:]-=3
        r=quality(g,b,a)
        self.assertAlmostEqual(r['groups']['all']['psnr_db'],10*math.log10(255**2/(19/3)))
        self.assertAlmostEqual(r['groups']['P']['psnr_db'],10*math.log10(255**2/9))

    def test_absent_mask_is_not_background_metric(self):
        a=np.full((2,2,2,3),32,dtype='uint8')
        r=quality(a,a-1,a-2)
        self.assertIsNone(r['groups']['P']['fish_psnr_db'])
        self.assertEqual(r['groups']['P']['fish_pixels'],0)

    def test_roi_uses_only_masked_pixels(self):
        g=np.full((2,2,2,3),32,dtype='uint8');a=g-1;mask=np.zeros((2,2,2),dtype='uint8');mask[:,0,0]=1
        a[:,0,0]=30
        r=quality(g,g-1,a,mask)
        self.assertEqual(r['groups']['all']['fish_pixels'],2)
        self.assertAlmostEqual(r['groups']['all']['fish_psnr_db'],10*math.log10(255**2/4))


if __name__=='__main__':unittest.main()
