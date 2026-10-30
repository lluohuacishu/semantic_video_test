"""Graph alignment cannot invent labels or silently remove real computation."""
import unittest
from analyze_profile import project,validate_fill_deletion


class GraphValidationTests(unittest.TestCase):
    def test_launch_shape_difference_cannot_be_projected(self):
        def kernel(grid):
            return dict(name='k',streamId=7,start=1,end=2,gridX=grid,gridY=1,gridZ=1,blockX=32,blockY=1,blockZ=1,
                        registersPerThread=8,staticSharedMemory=0,dynamicSharedMemory=0)
        with self.assertRaises(ValueError):project({'kernels':[kernel(1)]},{'kernels':[kernel(2)],'cpu_operators':[]})

    def test_ablation_cannot_silently_drop_real_computation(self):
        row=dict(name='conv',operator='aten::cudnn_convolution',streamId=7,start=1,end=2,gridX=1,gridY=1,gridZ=1,
                 blockX=32,blockY=1,blockZ=1,registersPerThread=8,staticSharedMemory=0,dynamicSharedMemory=0)
        with self.assertRaises(ValueError):validate_fill_deletion([row],[])
        removed=validate_fill_deletion([dict(row,name='fill',operator='aten::fill_'),row],[row])
        self.assertEqual(len(removed),1)


if __name__=='__main__':unittest.main()
