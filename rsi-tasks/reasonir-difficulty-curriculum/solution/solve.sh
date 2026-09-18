#!/bin/bash
set -euo pipefail

cd /workspace
rm -rf submission.next submission.previous
mkdir -p policy/reference
ln -sfn /opt/reasonir-task/base/ReasonIR-8B policy/reference/checkpoint
ln -sfn /opt/reasonir-task/base/ReasonIR-8B/tokenizer.json policy/reference/tokenizer.json
ln -sfn /opt/reasonir-task/base/ReasonIR-8B/modeling_reasonir_8b.py policy/reference/modeling_reasonir_8b.py
mkdir -p submission.next/adapter
ln -s /opt/reasonir-task/base/zero-adapter/adapter_model.safetensors submission.next/adapter/adapter_model.safetensors
ln -s /opt/reasonir-task/base/zero-adapter/adapter_config.json submission.next/adapter/adapter_config.json

python -c 'import hashlib,json,pathlib; root=pathlib.Path("submission.next"); adapter=pathlib.Path("/opt/reasonir-task/base/zero-adapter/adapter_model.safetensors"); config=pathlib.Path("/opt/reasonir-task/base/zero-adapter/adapter_config.json"); manifest={"schema_version":1,"baseline":True,"base_revision":"c3d0690370ff4a8c3d3882d8dfa85c43650034fa","adapter_file":"adapter/adapter_model.safetensors","adapter_sha256":hashlib.sha256(adapter.read_bytes()).hexdigest(),"adapter_config_file":"adapter/adapter_config.json","adapter_config_sha256":hashlib.sha256(config.read_bytes()).hexdigest(),"lora":{"r":16,"lora_alpha":64,"lora_dropout":0.1,"inference_mode":False,"target_modules":["q_proj","o_proj","v_proj","k_proj","w1","w2","w3"],"bias":"none","task_type":"FEATURE_EXTRACTION"}}; (root/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n")'

rm -rf submission
mv submission.next submission
