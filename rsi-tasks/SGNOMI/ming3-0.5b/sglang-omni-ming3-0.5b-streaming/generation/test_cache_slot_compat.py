"""Exercise the archived factory with the current read-only component contract."""
import ast
import dataclasses
import enum
from pathlib import Path
import tarfile
from types import SimpleNamespace
import unittest

class ComponentType(enum.Enum):
    FULL = 1
    SWA = 2
    MAMBA = 3

class Allocator:
    size = 4096

class SWAAllocator(Allocator):
    pass

class PureSWAAllocator(Allocator):
    pass

class HybridRequests:
    pass

@dataclasses.dataclass
class Params:
    disable: bool
    req_to_token_pool: object
    token_to_kv_pool_allocator: object
    page_size: int
    sliding_window_size: int | None
    chunked_prefill_size: int
    eviction_policy: str
    eviction_policy_config: object = None
    draft_swa_window: int = 0
    tree_slots: object = None

    @property
    def tree_components(self):
        return tuple(ComponentType[slot.component] for slot in self.tree_slots)

def factory(*, disabled=True, session=False, slots=()):
    archive = Path(__file__).resolve().parents[1] / 'environment/source.tar.gz'
    with tarfile.open(archive) as source:
        member = next(m for m in source if m.name.endswith('/sglang_omni/scheduling/sglang_backend/cache.py'))
        tree = ast.parse(source.extractfile(member).read().decode())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'create_tree_cache')
    module = ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',
                 names=[ast.alias(name='annotations')], level=0), function], type_ignores=[]))
    calls = {}
    def storage(requests, allocator, *, shape):
        calls.update(requests=requests, allocator=allocator, shape=shape)
        def tree_slots(*, window_reach):
            calls['window_reach'] = window_reach
            return slots
        return SimpleNamespace(tree_slots=tree_slots)
    values = dict(dataclasses=dataclasses, CacheInitParams=Params, ComponentType=ComponentType,
        HybridReqToTokenPool=HybridRequests, PureSWATokenToKVPoolAllocator=PureSWAAllocator,
        SWATokenToKVPoolAllocator=SWAAllocator, StaticPoolShape=lambda **kw:SimpleNamespace(**kw),
        kv_storage_of=storage, mamba_cache_chunk_size=lambda:64,
        get_memory=lambda:SimpleNamespace(disable_radix_cache=disabled,radix_eviction_policy='lru',radix_eviction_policy_config=None),
        get_schedule=lambda:SimpleNamespace(chunked_prefill_size=512),
        get_serving=lambda:SimpleNamespace(enable_streaming_session=session),
        UnifiedRadixCache=lambda params:params, PureSWARadixCache=lambda params:('pure',params),
        EvictHeapRadixCache=lambda params:('lru',params), RadixCache=lambda params:('other',params))
    exec(compile(module,str(archive),'exec'),values)
    return values['create_tree_cache'], calls

class CacheSlotCompatibilityTest(unittest.TestCase):
    def test_disabled_cache_uses_storage_owned_full_slots(self):
        slots=(SimpleNamespace(component='FULL'),)
        build,calls=factory(slots=slots)
        requests,allocator=object(),Allocator()
        params=build(requests,allocator,1,None)
        self.assertIs(params.tree_slots,slots)
        self.assertEqual(params.tree_components,(ComponentType.FULL,))
        self.assertIs(calls['requests'],requests)
        self.assertIs(calls['allocator'],allocator)
        self.assertFalse(calls['shape'].has_window)
        self.assertFalse(calls['shape'].has_state)
        self.assertEqual(calls['shape'].token_capacity,4096)

    def test_swa_and_state_shape_preserve_window_and_checkpoint_interval(self):
        slots=tuple(SimpleNamespace(component=n) for n in ('FULL','SWA','MAMBA'))
        build,calls=factory(slots=slots)
        params=build(HybridRequests(),SWAAllocator(),16,256)
        self.assertEqual(params.tree_components,tuple(ComponentType))
        self.assertTrue(calls['shape'].has_window and calls['shape'].has_state)
        self.assertEqual(calls['shape'].window,256)
        self.assertEqual(calls['shape'].checkpoint_interval,64)
        self.assertEqual(calls['window_reach'],0)

    def test_streaming_session_also_uses_storage_slots(self):
        slots=(SimpleNamespace(component='FULL'),)
        build,calls=factory(disabled=False,session=True,slots=slots)
        params=build(object(),Allocator(),1,None)
        self.assertFalse(params.disable)
        self.assertIs(params.tree_slots,slots)

if __name__=='__main__':
    unittest.main()
