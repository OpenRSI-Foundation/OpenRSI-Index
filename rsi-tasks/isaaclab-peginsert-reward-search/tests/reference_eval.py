#!/usr/bin/env python3
"""Maintainer-only released-reference measurement; never writes ordinary reward."""
import importlib.util,sys,traceback
spec=importlib.util.spec_from_file_location("ordinary_evaluator","/tests/evaluate.py")
if spec is None or spec.loader is None: raise SystemExit("cannot load fixed evaluator")
module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
try: raise SystemExit(module.run("reference"))
except Exception: traceback.print_exc(); raise SystemExit(1)
