"""Exercise a CUDA context and GPU memory, without downloading model weights."""

import ctypes as c
import json

cuda = c.CDLL("libcuda.so.1")


def call(name, types, *args):
    function = getattr(cuda, name)
    function.argtypes = types
    function.restype = c.c_int
    code = function(*args)
    assert code == 0, (name, code)


call("cuInit", [c.c_uint], 0)
count = c.c_int()
call("cuDeviceGetCount", [c.POINTER(c.c_int)], c.byref(count))
assert count.value == 1
context = c.c_void_p()
call(
    "cuCtxCreate_v2", [c.POINTER(c.c_void_p), c.c_uint, c.c_int], c.byref(context), 0, 0
)
pointer = c.c_uint64()
try:
    call("cuMemAlloc_v2", [c.POINTER(c.c_uint64), c.c_size_t], c.byref(pointer), 4)
    try:
        original, restored = c.c_float(2.0), c.c_float()
        call(
            "cuMemcpyHtoD_v2",
            [c.c_uint64, c.c_void_p, c.c_size_t],
            pointer,
            c.byref(original),
            4,
        )
        call(
            "cuMemcpyDtoH_v2",
            [c.c_void_p, c.c_uint64, c.c_size_t],
            c.byref(restored),
            pointer,
            4,
        )
        assert restored.value == 2.0
        print(
            json.dumps(
                {
                    "cuda_context": "passed",
                    "gpu_memory_roundtrip": restored.value,
                    "physical_gpu_count": count.value,
                }
            )
        )
    finally:
        call("cuMemFree_v2", [c.c_uint64], pointer)
finally:
    call("cuCtxDestroy_v2", [c.c_void_p], context)
