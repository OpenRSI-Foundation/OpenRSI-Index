"""Editable reference predictor. Judge supplies inputs only, never labels."""
import json
import os
from pathlib import Path

BASE_DIR = "/opt/kev-assets/base"


def clean_request(request):
    if not isinstance(request, dict) or set(request) != {"state", "questions"}:
        raise ValueError("prediction accepts only state and questions")
    if not isinstance(request["questions"], dict) or not request["questions"]:
        raise ValueError("questions must be a nonempty object")
    for question in request["questions"].values():
        if not isinstance(question, dict) or set(question) - {"type", "instructions", "criteria"}:
            raise ValueError("prediction questions may contain only type, instructions, and criteria")
    return request


class Predictor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def predict(self, request):
        request = clean_request(request)
        from kev.api import SystemOneRequest, to_record
        import torch
        # to_record creates dummy zeros required by encode; it never reads answers.
        record, metadata = to_record(SystemOneRequest(**request))
        with torch.inference_mode():
            probabilities = self.model.probs(self.model.encode(self.tokenizer, record, strict=True))
        result = {}
        for values, question in zip(probabilities, metadata, strict=True):
            keys = (question["keys"] if question["type"] == "choice" else
                    ["false", "true"] if question["type"] == "noul" else list(question["legend"]))
            result[question["id"]] = {key: float(value) for key, value in zip(keys, values, strict=True)}
        return {"probabilities": result}


def load(checkpoint_dir: str, device: str):
    checkpoint = Path(checkpoint_dir)
    for name in ("manifest.json", "head.pt", "adapter_config.json", "adapter_model.safetensors"):
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"missing {name}; train a complete candidate in Work before Judge evaluation")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from peft import PeftModel
    from kev.model import DecisionModel, load_tokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    meta = torch.load(checkpoint / "head.pt", map_location="cpu", weights_only=True)
    # Base/tokenizer paths are fixed, even when candidate metadata names another path.
    tokenizer = load_tokenizer(BASE_DIR)
    model = DecisionModel(BASE_DIR, tokenizer, device, lora=None, attn="eager",
                          head_dim=meta.get("head_dim", 256),
                          option_isolation=meta.get("option_isolation", False), dtype=torch.float32)
    model.lm = PeftModel.from_pretrained(model.lm, str(checkpoint), local_files_only=True).to(device)
    adapter = json.loads((checkpoint / "adapter_config.json").read_text())
    if not adapter.get("trainable_token_indices"):
        model.lm = model.lm.merge_and_unload()
    model.head.load_state_dict(meta["head"])
    model.eval()
    return Predictor(tokenizer, model)
