"""Guard experimental comparison against false correctness and invalid numerics."""
import copy
import unittest
import numpy as np
from experiment_common import difference
from run_experiment import compare


class ValidationTests(unittest.TestCase):
    def row(self):
        return dict(records=[dict(index=0,kind=0,latent_sha256='a',payload_sha256='b',
                                  entropy=[{'symbols_sha256':'c','cdf_sha256':'d'}],rgb_sha256='e')])

    def test_equal(self):
        a=np.array([1.,2.],dtype='float32')
        self.assertTrue(difference(a,a.copy())['bit_exact'])
        compare(self.row(),self.row(),rgb=True)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):difference(np.zeros(1),np.array([np.nan]))

    def test_shape_rejected(self):
        with self.assertRaises(ValueError):difference(np.zeros(1),np.zeros(2))

    def test_symbol_or_cdf_change_rejected(self):
        for key in ['symbols_sha256','cdf_sha256']:
            a,b=self.row(),self.row();b['records'][0]['entropy'][0][key]='changed'
            with self.assertRaises(ValueError):compare(a,b)

    def test_reference_and_rgb_change_rejected(self):
        for key in ['latent_sha256','payload_sha256','rgb_sha256']:
            a,b=self.row(),self.row();b['records'][0][key]='changed'
            with self.assertRaises(ValueError):compare(a,b,rgb=True)

    def test_frame_count_rejected(self):
        a,b=self.row(),self.row();b['records'].clear()
        with self.assertRaises(ValueError):compare(a,b)


if __name__=='__main__':unittest.main()
