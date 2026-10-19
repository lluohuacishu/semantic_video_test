"""Explicit opt-in ctypes backend; original Python GaussianRans remains unchanged."""
import ctypes as C
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parent.parent / '04_外部依赖快照/semantic-video-training/scripts'
sys.path.insert(0, str(SCRIPTS))
from glc_entropy import GaussianRans, MAX_SYMBOLS


class NativeGaussianRans(GaussianRans):
    def __init__(self, library, reference=None):
        if reference is None:
            super().__init__()
        else:
            for name in ('scales', 'log_min', 'log_step', 'tables', 'radii', 'table_sha256'):
                setattr(self, name, getattr(reference, name))
        self.library_path = Path(library).resolve(strict=True)
        self.lib = C.CDLL(str(self.library_path))
        self.lib.glc_rans_abi.restype = C.c_int
        self.lib.glc_rans_abi.argtypes = []
        if self.lib.glc_rans_abi() != 1:
            raise RuntimeError('Unsupported native rANS ABI')
        self._cdf = np.ascontiguousarray([v for row in self.tables for v in row], dtype=np.uint32)
        self._offset = np.ascontiguousarray(np.r_[0, np.cumsum([len(row) for row in self.tables])], dtype=np.uint32)
        self._radius = np.ascontiguousarray(self.radii, dtype=np.int32)
        ptr, size = C.c_void_p, C.c_size_t
        self.lib.glc_rans_encode.argtypes = [ptr, ptr, size, ptr, size, ptr, ptr, ptr, size, C.POINTER(size)]
        self.lib.glc_rans_encode.restype = C.c_int
        self.lib.glc_rans_decode.argtypes = [ptr, size, ptr, size, ptr, size, ptr, ptr, ptr, size]
        self.lib.glc_rans_decode.restype = C.c_int
        for array in (self._cdf, self._offset, self._radius):
            array.flags.writeable = False

    def _native_indexes(self, indexes):
        values = np.asarray(indexes).reshape(-1)
        if (len(values) > MAX_SYMBOLS or not np.isfinite(values).all() or
                np.any(values != np.rint(values)) or np.any(values < 0) or np.any(values >= len(self.tables))):
            raise ValueError('Invalid entropy table indexes.')
        return np.ascontiguousarray(values, dtype=np.uint16)

    def _tables(self):
        return (self._cdf.ctypes.data, self._cdf.size, self._offset.ctypes.data, self._radius.ctypes.data)

    def encode(self, symbols, indexes):
        indexes = self._native_indexes(indexes)
        values = np.asarray(symbols).reshape(-1)
        if len(values) != len(indexes) or not np.isfinite(values).all():
            raise ValueError('Symbols and probability indexes do not match.')
        if np.any(values < -(2**31)) or np.any(values >= 2**31) or np.any(values != np.rint(values)):
            raise ValueError('Entropy symbols must be signed 32-bit integers.')
        values = np.ascontiguousarray(values, dtype=np.int32)
        buffer = np.empty(16 + 7 * len(values), dtype=np.uint8)
        written = C.c_size_t()
        status = self.lib.glc_rans_encode(values.ctypes.data, indexes.ctypes.data, len(values),
                                        *self._tables(), buffer.ctypes.data, len(buffer), C.byref(written))
        if status:
            raise ValueError(f'Native entropy encode rejected input (status {status}).')
        return buffer[:written.value].tobytes()

    def decode(self, stream, indexes):
        indexes = self._native_indexes(indexes)
        # Copy writable/noncontiguous callers; own immutable bytes throughout the GIL-releasing call.
        data = np.frombuffer(bytes(stream), dtype=np.uint8)
        result = np.empty(len(indexes), dtype=np.int32)
        status = self.lib.glc_rans_decode(data.ctypes.data, len(data), indexes.ctypes.data, len(indexes),
                                        *self._tables(), result.ctypes.data, len(result))
        if status:
            raise ValueError(f'Native entropy decode rejected stream (status {status}).')
        return result
