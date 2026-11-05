"""Compare literal Omni setup calls with actual installed core declarations."""
import ast
import builtins
import copy
import importlib.metadata
import inspect
import json
from pathlib import Path
import symtable

core = Path(importlib.metadata.distribution('sglang').locate_file('sglang/srt'))
omni = Path('/opt/rsi-source/sglang_omni')
specifications = {
    'SchedulerDPAttnAdapter': ('managers/scheduler_components/dp_attn.py', None),
    'SchedulerPoolStatsObserver': ('managers/scheduler_components/pool_stats_observer.py', None),
    'SchedulerLoadInquirer': ('managers/scheduler_components/load_inquirer.py', None),
    'SchedulerBatchResultProcessor': ('managers/scheduler_components/batch_result_processor.py', None),
    'SchedulerLogprobResultProcessor': ('managers/scheduler_components/logprob_result_processor.py', None),
    'KvWeightVersionTracker.maybe_create': ('mem_cache/kv_weight_version_tracker.py', 'maybe_create'),
    'StaticPoolShape': ('mem_cache/static_kv_storage.py', None),
    'kv_storage_of': ('mem_cache/static_kv_storage.py', None),
    'CacheInitParams': ('mem_cache/cache_init_params.py', None),
    'UnifiedRadixCache': ('mem_cache/unified_radix_cache.py', None),
    'RequestList': ('managers/scheduler_components/request_list.py', None),
    'RecentPrefillBatchSizeTracker': ('managers/prefill_delayer.py', None),
}

def signature(name):
    path, method = specifications[name]
    tree = ast.parse((core / path).read_text())
    wanted = name.split('.')[0]
    node = next(n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == wanted)
    if isinstance(node, ast.ClassDef):
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == (method or '__init__')), None)
        if init is None:
            fields = []
            for n in node.body:
                if not isinstance(n, ast.AnnAssign) or not isinstance(n.target, ast.Name):
                    continue
                if isinstance(n.value, ast.Call) and any(k.arg == 'init' and isinstance(k.value, ast.Constant) and k.value.value is False for k in n.value.keywords):
                    continue
                fields.append(n)
            return [n.target.id for n in fields], {n.target.id for n in fields if n.value is None}, False
        node = init
    positional = node.args.posonlyargs + node.args.args
    optional = {n.arg for n in positional[len(positional)-len(node.args.defaults):]} if node.args.defaults else set()
    fields = [n.arg for n in positional + node.args.kwonlyargs if n.arg not in ('self', 'cls')]
    required = {n.arg for n in positional if n.arg not in optional and n.arg not in ('self', 'cls')}
    required.update(n.arg for n, default in zip(node.args.kwonlyargs, node.args.kw_defaults) if default is None)
    return fields, required, node.args.kwarg is not None

checks = []
omni_tree = ast.parse((omni / 'scheduling/omni_scheduler.py').read_text())
core_tree = ast.parse((core / 'managers/scheduler.py').read_text())
omni_class = next(n for n in omni_tree.body if isinstance(n, ast.ClassDef) and n.name == 'OmniScheduler')
core_class = next(n for n in core_tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
own_methods = {n.name: n for n in omni_class.body if isinstance(n, ast.FunctionDef)}
core_methods = {n.name: n for n in core_class.body if isinstance(n, ast.FunctionDef)}

def shape_signature(method):
    arguments = copy.deepcopy(method.args)
    for argument in arguments.posonlyargs + arguments.args + arguments.kwonlyargs:
        argument.annotation = None
    for argument in (arguments.vararg, arguments.kwarg):
        if argument:
            argument.annotation = None
    arguments.defaults = [ast.Constant(None) for _ in arguments.defaults]
    arguments.kw_defaults = [None if default is None else ast.Constant(None) for default in arguments.kw_defaults]
    stub = ast.FunctionDef(name='shape', args=arguments, body=[ast.Pass()], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[]))
    namespace = {}
    exec(compile(module, '<signature-only>', 'exec'), namespace)
    return inspect.signature(namespace['shape'])

pending, visited, override_checks = [], set(), []
for node in ast.walk(omni_class):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        owner, name = node.func.value.id, node.func.attr
        if owner == '_Upstream' or (owner == 'self' and name not in own_methods):
            pending.append(name)
while pending:
    name = pending.pop()
    if name in visited or name not in core_methods:
        continue
    visited.add(name)
    for node in ast.walk(core_methods[name]):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
            continue
        callee = node.func.attr
        if callee not in own_methods:
            pending.append(callee)
            continue
        if any(isinstance(arg, ast.Starred) for arg in node.args) or any(k.arg is None for k in node.keywords):
            continue
        try:
            shape_signature(own_methods[callee]).bind(None, *[None for _ in node.args],
                                                     **{k.arg: None for k in node.keywords})
        except TypeError as error:
            raise AssertionError(dict(core_method=name, core_line=node.lineno,
                                      omni_override=callee, error=str(error))) from error
        override_checks.append(dict(core_method=name, omni_override=callee, core_line=node.lineno))
global_checks = []
import_checks = []
asr_model = ast.parse((omni / 'models/qwen3_asr/sglang_model.py').read_text())
asr_prepare = next(n for n in asr_model.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'fused_asr_forward_prepare_native')
qwen3_tree = ast.parse((core / 'models/qwen3.py').read_text())
attention = next(n for n in qwen3_tree.body if isinstance(n, ast.ClassDef)
                 and n.name == 'Qwen3Attention')
native_prepare = next(n for n in attention.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'forward_prepare_native')
attention_checks = []
for owner, method_name, signature_method in (
        (attention, 'forward_prepare_native', asr_prepare),
        (asr_prepare, '_asr_unfused_forward_prepare_native', native_prepare)):
    for call in ast.walk(owner):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == method_name):
            continue
        assert not any(isinstance(arg, ast.Starred) for arg in call.args)
        assert all(k.arg is not None for k in call.keywords)
        shape_signature(signature_method).bind(None, *[None for _ in call.args],
                                               **{k.arg: None for k in call.keywords})
        attention_checks.append(dict(method=method_name, line=call.lineno))
assert len(attention_checks) >= 2, 'Audit core-to-ASR and ASR-to-native callback calls'
audited_files = ('model_runner/sglang_model_runner.py', 'scheduling/omni_scheduler.py',
                 'scheduling/sglang_backend/cache.py') + tuple(
    'models/qwen3_asr/' + name + '.py' for name in (
        'stages', 'engine_builder', 'request_builders', 'encoder_service',
        'configuration_qwen3_asr', 'sglang_model', 'encoder_cuda_graph',
        'encoder_npu_graph', 'metadata_glue', 'mrope_fast_path', 'audio_lengths'))
for relative in audited_files:
    source = (omni / relative).read_text()
    table = symtable.symtable(source, relative, 'exec')
    defined = {s.get_name() for s in table.get_symbols()
               if s.is_assigned() or s.is_imported() or s.is_namespace()}
    defined.update(dir(builtins))
    defined.update(('__name__', '__file__', '__doc__', '__package__', '__spec__', '__builtins__'))
    pending, missing = [table], []
    while pending:
        scope = pending.pop()
        pending.extend(scope.get_children())
        missing.extend((scope.get_name(), scope.get_lineno(), s.get_name())
                       for s in scope.get_symbols()
                       if s.is_global() and s.is_referenced() and s.get_name() not in defined)
    assert not missing, dict(file=relative, undefined_globals=missing)
    global_checks.append(relative)
    for node in ast.walk(ast.parse(source)):
        if not (isinstance(node, ast.ImportFrom) and node.module
                and node.module.startswith('sglang.') and not node.level):
            continue
        module_path = core.parent.joinpath(*node.module.split('.')[1:])
        candidates = (module_path.with_suffix('.py'), module_path / '__init__.py')
        module_file = next((p for p in candidates if p.is_file()), None)
        if module_file is None:
            continue  # Compiled extension; the runtime imports check remains authoritative.
        exported = symtable.symtable(module_file.read_text(), str(module_file), 'exec')
        names = {s.get_name() for s in exported.get_symbols()
                 if s.is_assigned() or s.is_imported() or s.is_namespace()}
        if module_path.is_dir():
            # Python's from-package import also resolves child modules.
            names.update(p.name if p.is_dir() else p.name.split('.')[0]
                         for p in module_path.iterdir()
                         if p.is_dir() or p.suffix in ('.py', '.so'))
        if '__getattr__' in names:
            continue  # Dynamic exports are checked by the runtime import phase.
        missing = {alias.name for alias in node.names if alias.name != '*'} - names
        if missing:
            # Re-exports through import-star are absent from the symbol table.
            actual_module = importlib.import_module(node.module)
            missing = {name for name in missing if not hasattr(actual_module, name)}
        assert not missing, dict(file=relative, line=node.lineno, module=node.module,
                                 absent_module_exports=sorted(missing))
        import_checks.append(dict(file=relative, line=node.lineno, module=node.module))
for relative in ('scheduling/omni_scheduler.py', 'scheduling/sglang_backend/cache.py'):
    tree = ast.parse((omni / relative).read_text())
    if relative.endswith('omni_scheduler.py'):
        selected = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in
                    ('init_upstream_scheduler_components', 'init_upstream_compat_flags')]
        assert len(selected) == 2
        tree = ast.Module(body=selected, type_ignores=[])
    dictionaries = {}
    optional_dictionary_keys = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
            assert all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.value.keys), (relative, target.id, 'dynamic dictionary keys')
            assert target.id not in dictionaries, (relative, target.id, 'ambiguous dictionary binding')
            dictionaries[target.id] = {k.value for k in node.value.keys}
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str):
            optional_dictionary_keys.setdefault(target.value.id, set()).add(target.slice.value)
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call):
            continue
        name = ast.unparse(call.func)
        if name not in specifications:
            continue
        fields, required, arbitrary_keywords = signature(name)
        assert not any(isinstance(n, ast.Starred) for n in call.args), (relative, name, 'dynamic positional args')
        given = set(fields[:len(call.args)]) | {k.arg for k in call.keywords if k.arg is not None}
        optional = set()
        for keyword in call.keywords:
            if keyword.arg is None:
                assert isinstance(keyword.value, ast.Name) and keyword.value.id in dictionaries, (relative, name, 'unresolved keyword mapping')
                given.update(dictionaries[keyword.value.id])
                optional.update(optional_dictionary_keys.get(keyword.value.id, set()))
        unsupported = set() if arbitrary_keywords else (given | optional) - set(fields)
        missing = required - given
        assert not unsupported and not missing, dict(file=relative, component=name, unsupported=sorted(unsupported), missing=sorted(missing))
        checks.append(dict(file=relative, component=name, line=call.lineno))
assert len(checks) >= 10, 'The intended setup calls were not covered'
print(json.dumps(dict(status='passed', actual_core_root=str(core), checks=checks,
                      undefined_global_checks=global_checks, override_checks=override_checks,
                      module_export_checks=import_checks,
                      attention_callback_checks=attention_checks,
                      borrowed_methods_audited=sorted(visited))))
