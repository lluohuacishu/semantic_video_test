"""Same native cold decoder and audit; use faster lossless PNG compression only."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
PROJECT=Path(r"C:\Users\lenovo\Desktop\works\work\semantic-video-mlvc")
sys.path.insert(0,str(PROJECT/"adapters"))
import mlvc_file


def save_rgb(path,pixels):Image.fromarray(pixels).save(path,compress_level=1)


if __name__=="__main__":
    mlvc_file.save_rgb=save_rgb
    mlvc_file.main()
