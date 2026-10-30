"""Small CUPTI Activity API client for the installed CUDA 12.4/API-22 DLL.

ABI names/layouts follow NVIDIA's public CUPTI Activity API (Kernel9, Memcpy5,
Memset4). No CUDA toolkit install, environment upgrade, or counter access.
https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html
"""
from __future__ import annotations

import ctypes as C
from pathlib import Path
import threading

U8, U16, U32, I32, U64, I64, PTR = C.c_uint8, C.c_uint16, C.c_uint32, C.c_int32, C.c_uint64, C.c_int64, C.c_void_p


class API(C.Structure):
    _fields_ = [('kind', U32), ('cbid', U32), ('start', U64), ('end', U64),
                ('processId', U32), ('threadId', U32), ('correlationId', U32), ('returnValue', U32)]


class External(C.Structure):
    _fields_ = [('kind', U32), ('externalKind', U32), ('externalId', U64), ('correlationId', U32), ('reserved', U32)]


class Kernel9(C.Structure):
    _fields_ = [('kind', U32), ('cacheConfig', U8), ('sharedMemoryConfig', U8), ('registersPerThread', U16),
                ('partitionedGlobalCacheRequested', U32), ('partitionedGlobalCacheExecuted', U32),
                ('start', U64), ('end', U64), ('completed', U64), ('deviceId', U32), ('contextId', U32), ('streamId', U32),
                *[(name, I32) for name in ['gridX','gridY','gridZ','blockX','blockY','blockZ','staticSharedMemory','dynamicSharedMemory']],
                ('localMemoryPerThread', U32), ('localMemoryTotal', U32), ('correlationId', U32),
                ('gridId', I64), ('name', PTR), ('reserved0', PTR), ('queued', U64), ('submitted', U64),
                ('launchType', U8), ('isSharedMemoryCarveoutRequested', U8), ('sharedMemoryCarveoutRequested', U8), ('padding', U8),
                ('sharedMemoryExecuted', U32), ('graphNodeId', U64), ('shmemLimitConfig', U32), ('graphId', U32),
                ('pAccessPolicyWindow', PTR), ('channelID', U32), ('channelType', U32),
                ('clusterX', U32), ('clusterY', U32), ('clusterZ', U32), ('clusterSchedulingPolicy', U32),
                ('localMemoryTotal_v2', U64), ('maxPotentialClusterSize', U32), ('maxActiveClusters', U32)]


class Memcpy5(C.Structure):
    _fields_ = [('kind', U32), ('copyKind', U8), ('srcKind', U8), ('dstKind', U8), ('flags', U8),
                ('bytes', U64), ('start', U64), ('end', U64), ('deviceId', U32), ('contextId', U32),
                ('streamId', U32), ('correlationId', U32), ('runtimeCorrelationId', U32), ('pad', U32),
                ('reserved0', PTR), ('graphNodeId', U64), ('graphId', U32), ('channelID', U32), ('channelType', U32), ('pad2', U32)]


class Memset4(C.Structure):
    _fields_ = [('kind', U32), ('value', U32), ('bytes', U64), ('start', U64), ('end', U64),
                ('deviceId', U32), ('contextId', U32), ('streamId', U32), ('correlationId', U32),
                ('flags', U16), ('memoryKind', U16), ('pad', U32), ('reserved0', PTR),
                ('graphNodeId', U64), ('graphId', U32), ('channelID', U32), ('channelType', U32), ('pad2', U32)]


class Synchronization(C.Structure):
    _fields_ = [('kind', U32), ('type', U32), ('start', U64), ('end', U64),
                ('correlationId', U32), ('contextId', U32), ('streamId', U32), ('cudaEventId', U32)]


class CudaEvent(C.Structure):
    _fields_ = [('kind', U32), ('correlationId', U32), ('contextId', U32), ('streamId', U32), ('eventId', U32), ('pad', U32)]


class Marker2(C.Structure):
    _fields_ = [('kind',U32),('flags',U32),('timestamp',U64),('id',U32),('objectKind',U32),
                ('objectId',U32*3),('pad',U32),('name',PTR),('domain',PTR)]


class MarkerData(C.Structure):
    _fields_ = [('kind',U32),('flags',U32),('id',U32),('payloadKind',U32),('payload',U64),('color',U32),('category',U32)]


class CuptiTrace:
    KINDS = (1, 2, 4, 5, 10, 38, 39)
    CUSTOM0 = 3

    def __init__(self, dll=None, kinds=None):
        import torch
        dll = Path(dll) if dll else Path(torch.__file__).parent / 'lib/cupti64_2024.1.0.dll'
        self.dll = str(dll)
        self.kinds=tuple(kinds or self.KINDS)
        self.lib = C.WinDLL(str(dll))
        signatures = {
            'cuptiGetVersion': ([C.POINTER(U32)], I32),
            'cuptiGetResultString': ([I32, C.POINTER(C.c_char_p)], I32),
            'cuptiGetTimestamp': ([C.POINTER(U64)], I32),
            'cuptiActivityEnable': ([U32], I32), 'cuptiActivityDisable': ([U32], I32),
            'cuptiActivityFlushAll': ([U32], I32),
            'cuptiActivityGetNextRecord': ([PTR, C.c_size_t, C.POINTER(PTR)], I32),
            'cuptiActivityGetNumDroppedRecords': ([PTR, U32, C.POINTER(C.c_size_t)], I32),
            'cuptiActivityPushExternalCorrelationId': ([U32, U64], I32),
            'cuptiActivityPopExternalCorrelationId': ([U32, C.POINTER(U64)], I32),
            'cuptiGetCallbackName': ([U32, U32, C.POINTER(C.c_char_p)], I32),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = arguments, result
        version = U32()
        self.check(self.lib.cuptiGetVersion(C.byref(version)))
        self.version = version.value
        if self.version != 22 or C.sizeof(PTR) != 8:
            raise RuntimeError('Only the inspected CUPTI API 22 / 64-bit record ABI is supported')
        self.records, self.errors, self.buffers = [], [], {}
        self.dropped_records, self.enabled = 0, False
        self.lock = threading.RLock()
        self.request_type = C.WINFUNCTYPE(None, C.POINTER(PTR), C.POINTER(C.c_size_t), C.POINTER(C.c_size_t))
        self.complete_type = C.WINFUNCTYPE(None, PTR, U32, PTR, C.c_size_t, C.c_size_t)
        self.request_cb = self.request_type(self._request)
        self.complete_cb = self.complete_type(self._complete)
        self.lib.cuptiActivityRegisterCallbacks.argtypes = [self.request_type, self.complete_type]
        self.lib.cuptiActivityRegisterCallbacks.restype = I32
        self.check(self.lib.cuptiActivityRegisterCallbacks(self.request_cb, self.complete_cb))

    def result_name(self, code):
        name = C.c_char_p()
        self.lib.cuptiGetResultString(code, C.byref(name))
        return name.value.decode('utf-8') if name.value else str(code)

    def check(self, code):
        if code:
            raise RuntimeError(self.result_name(code))

    def _request(self, pointer, size, max_records):
        try:
            buffer = C.create_string_buffer(8 * 1024 * 1024)
            address = C.addressof(buffer)
            with self.lock:
                self.buffers[address] = buffer
            pointer[0], size[0], max_records[0] = address, len(buffer), 0
        except BaseException as exc:
            self.errors.append('request: ' + repr(exc))
            pointer[0], size[0], max_records[0] = None, 0, 0

    def _complete(self, context, stream, buffer, size, valid):
        try:
            pointer = PTR()
            while True:
                status = self.lib.cuptiActivityGetNextRecord(buffer, valid, C.byref(pointer))
                if status:
                    if self.result_name(status) != 'CUPTI_ERROR_MAX_LIMIT_REACHED':
                        self.errors.append('record: ' + self.result_name(status))
                    break
                kind = C.cast(pointer, C.POINTER(U32))[0]
                types = {1: Memcpy5, 2: Memset4, 4: API, 5: API, 10: Kernel9,12:Marker2,13:MarkerData, 36: CudaEvent, 38: Synchronization, 39: External}
                if kind not in types:
                    error='Unknown activity kind: ' + str(kind)
                    if error not in self.errors:self.errors.append(error)
                    continue
                value = C.cast(pointer, C.POINTER(types[kind])).contents
                fields = {name: getattr(value, name) for name, _ in value._fields_
                          if name not in ('name','domain','objectId','reserved0','pAccessPolicyWindow')}
                if kind in (10,12):
                    fields['name'] = C.string_at(value.name).decode('utf-8', errors='replace') if value.name else ''
                if kind==12:
                    fields['objectId']=list(value.objectId)
                    fields['domain']=C.string_at(value.domain).decode('utf-8',errors='replace') if value.domain else ''
                if kind in (4, 5):
                    name = C.c_char_p()
                    code = self.lib.cuptiGetCallbackName(1 if kind == 4 else 2, value.cbid, C.byref(name))
                    fields['name'] = name.value.decode('utf-8', errors='replace') if code == 0 and name.value else f'cbid_{value.cbid}'
                self.records.append(fields)
            dropped = C.c_size_t()
            self.check(self.lib.cuptiActivityGetNumDroppedRecords(context, stream, C.byref(dropped)))
            self.dropped_records += dropped.value
        except BaseException as exc:
            self.errors.append('complete: ' + repr(exc))
        finally:
            with self.lock:
                self.buffers.pop(buffer, None)

    def timestamp(self):
        value = U64()
        self.check(self.lib.cuptiGetTimestamp(C.byref(value)))
        return value.value

    def start(self):
        for kind in self.kinds:
            self.check(self.lib.cuptiActivityEnable(kind))
        self.enabled = True

    def push(self, value):
        self.check(self.lib.cuptiActivityPushExternalCorrelationId(self.CUSTOM0, value))

    def pop(self, expected):
        value = U64()
        self.check(self.lib.cuptiActivityPopExternalCorrelationId(self.CUSTOM0, C.byref(value)))
        if value.value != expected:
            raise RuntimeError('CUPTI external-ID stack mismatch')

    def flush(self):
        self.check(self.lib.cuptiActivityFlushAll(0))
        if self.errors or self.dropped_records:
            raise RuntimeError(f'Incomplete CUPTI capture: errors={self.errors}, dropped={self.dropped_records}')

    def close(self):
        if self.enabled:
            for kind in self.kinds:
                self.check(self.lib.cuptiActivityDisable(kind))
            self.enabled = False
        self.flush()

    def take(self):
        self.flush()
        records, self.records = self.records, []
        return records


def union_ns(intervals):
    total, left, right = 0, None, None
    for begin, end in sorted(intervals):
        if end < begin:
            raise ValueError('Negative activity duration')
        if left is None:
            left, right = begin, end
        elif begin <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = begin, end
    return total + (right-left if left is not None else 0)
