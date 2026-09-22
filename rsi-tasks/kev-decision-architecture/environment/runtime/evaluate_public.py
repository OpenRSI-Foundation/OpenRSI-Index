"""Advisory Work-side evaluation on the fixed public development data only."""
import argparse
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys

PUBLIC_DATA = Path("/opt/kev-assets/public/development.jsonl")
_spec = importlib.util.spec_from_file_location("_kev_public_contract", Path(__file__).with_name("contract.py"))
contract = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(contract)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--candidate", default="/workspace/candidate")
    result.add_argument("--device", choices=("cpu", "cuda", "cuda:0"), default="cuda:0")
    result.add_argument("--limit", type=int, help="evaluate the first N public records; reports an explicit partial smoke result")
    return result


def evaluate_records(records, predict, limit=None):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    selected = records if limit is None else records[:limit]
    # Retain labelled records in the scorer; candidate gets a deep input-only copy.
    predictions = [predict(contract.clean_request(record)) for record in selected]
    return {
        "evaluation": "public_development", "formal": False,
        "partial": len(selected) < len(records), "limit": limit,
        "records": len(selected), "total_records": len(records),
        "scope": "advisory public metrics; this command does not invoke Judge or write a reward file",
        "metrics": contract.score_records(selected, predictions),
    }


def main(argv=None):
    options = parser()
    args = options.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        options.error("--limit must be positive")
    candidate = Path(args.candidate).absolute()
    problems = contract.candidate_errors(candidate)
    if problems:
        print(json.dumps({"status": "candidate_invalid", "formal": False, "errors": problems}, indent=2))
        return 1
    records = [json.loads(line) for line in PUBLIC_DATA.read_text().splitlines() if line.strip()]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    import torch
    torch.set_num_threads(8)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    sys.path.insert(0, str(candidate))
    spec = importlib.util.spec_from_file_location("_kev_public_candidate", candidate / "entrypoint.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    # Candidate logs remain visible on stderr; stdout is the aggregate JSON report.
    with contextlib.redirect_stdout(sys.stderr):
        spec.loader.exec_module(module)
        predictor = module.load(str(candidate / "checkpoint"), args.device)
        if not isinstance(predictor.model, torch.nn.Module):
            raise TypeError("entrypoint.load must expose a torch.nn.Module as .model")
        predictor.model.eval()
        with torch.inference_mode():
            report = evaluate_records(records, predictor.predict, args.limit)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
