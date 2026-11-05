"""The archived scheduler bridge must follow Python descriptor binding."""
import ast
from functools import cached_property
from pathlib import Path
import tarfile
import unittest

class Parent:
    @property
    def inherited_value(self):
        return self.value + 1

class Upstream(Parent):
    def add(self, amount):
        return self.value + amount

    @property
    def running_count(self):
        return len(self.running_reqs)

    @cached_property
    def frozen_count(self):
        return len(self.running_reqs)

    @staticmethod
    def utility(value):
        return value * 2

    @classmethod
    def owner_name(cls):
        return cls.__name__

def bridge_class():
    archive = Path(__file__).resolve().parents[1] / 'environment/source.tar.gz'
    with tarfile.open(archive) as source:
        member = next(m for m in source if m.name.endswith('/sglang_omni/scheduling/omni_scheduler.py'))
        tree = ast.parse(source.extractfile(member).read().decode())
    scheduler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OmniScheduler')
    method = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == '__getattr__')
    cls = ast.ClassDef(name='Bridge',bases=[],keywords=[],body=[method],decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[]))
    values = dict(_Upstream=Upstream)
    exec(compile(module,str(archive),'exec'),values)
    return values['Bridge']

class SchedulerDescriptorCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.bridge = bridge_class()()
        self.bridge.value = 3
        self.bridge.running_reqs = [object()]

    def test_normal_method_uses_omni_instance(self):
        self.assertEqual(self.bridge.add(4),7)

    def test_property_and_inherited_property_are_evaluated(self):
        self.assertEqual(self.bridge.inherited_value,4)
        self.assertEqual(self.bridge.running_count,1)
        self.bridge.running_reqs.append(object())
        self.assertEqual(self.bridge.running_count,2)

    def test_static_and_class_methods_are_not_double_bound(self):
        self.assertEqual(self.bridge.utility(5),10)
        self.assertEqual(self.bridge.owner_name(),'Upstream')

    def test_cached_property_retains_its_normal_semantics(self):
        self.assertEqual(self.bridge.frozen_count,1)
        self.bridge.running_reqs.append(object())
        self.assertEqual(self.bridge.frozen_count,1)

    def test_missing_attributes_raise_and_grammar_state_is_retained(self):
        with self.assertRaises(AttributeError):
            getattr(self.bridge,'missing')
        self.bridge.grammar_queue.append('pending')
        self.assertEqual(self.bridge.grammar_queue,['pending'])
        self.assertIsNone(self.bridge.grammar_backend)

if __name__=='__main__':
    unittest.main()
