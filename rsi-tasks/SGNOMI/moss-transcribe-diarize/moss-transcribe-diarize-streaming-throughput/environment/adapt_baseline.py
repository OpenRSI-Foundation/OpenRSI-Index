"""Apply common baseline compatibility and cold-input changes before calibration."""

import ast
from pathlib import Path
import sys


MODEL_CODE = 'moss_transcribe_diarize'
ASSETS = Path(__file__).resolve().parent


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError("Pinned source does not match the baseline adapter")
    else:
        return text.replace(old, new, 1)


def replace_bodies(path: Path, replacements: dict[str, str]) -> None:
    source = path.read_text()
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in replacements
    ]
    if {node.name for node in functions} != set(replacements):
        raise ValueError("A required baseline policy function is missing")
    else:
        pass
    for node in sorted(functions, key=lambda node: node.lineno, reverse=True):
        indent = " " * node.body[0].col_offset
        lines[node.body[0].lineno - 1:node.end_lineno] = [
            indent + replacements[node.name] + "\n"
        ]
    path.write_text("".join(lines))


def adapt_source(source_root: Path) -> None:
    package = source_root / "sglang_omni"
    model = package / "models" / MODEL_CODE
    replacements = {
        "lookup_cached_embedding": "return None",
        "cache_key": "return None",
        "cache_embedding": "return None",
        "retry_batch": "return False",
    }
    if MODEL_CODE == "whisper_asr":
        replacements.update(
            cache_key_from_fingerprint="return None",
            stage_host_copy="return None",
        )
        request_type = "WhisperASRRequestData"
        tokenizer_type = "WhisperTokenizer"
        decode_expression = "self.tokenizer.decode(token_ids, skip_special_tokens=True)"
    elif MODEL_CODE == "moss_transcribe_diarize":
        replacements.update(
            _lookup_cached_embedding="return None",
            recover_after_failure="return None",
            handle_batch_failure="return self.detach_failure(exc).exception",
            handle_item_failure="return self.detach_failure(exc).exception",
        )
        request_type = "MossTranscribeDiarizeRequestData"
        tokenizer_type = "PreTrainedTokenizerBase"
        decode_expression = (
            "postprocess_moss_transcribe_diarize_text(\n"
            "            decode_token_ids(self.tokenizer, token_ids, skip_special_tokens=False)\n"
            "        )"
        )
    else:
        raise ValueError("Unsupported derived model")
    replace_bodies(model / "encoder_service.py", replacements)
    (package / "scheduling" / "exact_text_streaming.py").write_text(
        (ASSETS / "exact_text_streaming.py").read_text()
    )
    request_path = model / "request_builders.py"
    request_source = request_path.read_text()
    request_source = replace_once(
        request_source, "from sglang_omni.proto import StagePayload\n",
        "from sglang_omni.proto import StagePayload\n"
        "from sglang_omni.scheduling.exact_text_streaming import ExactTextStreamState\n"
    )
    request_source = replace_once(
        request_source,
        "    engine_start_s: float = 0.0\n",
        "    engine_start_s: float = 0.0\n"
        "    exact_text_stream_state: ExactTextStreamState | None = None\n",
    )
    if MODEL_CODE == "whisper_asr":
        request_source = replace_once(
            request_source,
            "from sglang_omni.scheduling.sglang_backend import SGLangARRequestData\n",
            "from sglang_omni.scheduling.message import OutgoingMessage\n"
            "from sglang_omni.scheduling.sglang_backend import SGLangARRequestData\n"
            "from sglang_omni.scheduling.types import RequestOutput\n",
        )
    else:
        request_source = replace_once(
            request_source,
            "from sglang_omni.scheduling.token_text_streaming import (\n"
            "    make_token_text_stream_output_builder,\n)\n", "",
        )
    stream_builder = (
        (ASSETS / "stream_builder.template").read_text()
        .replace("TOKENIZER_TYPE", tokenizer_type)
        .replace("REQUEST_DATA_TYPE", request_type)
        .replace("DECODE_EXPRESSION", decode_expression)
    )
    if MODEL_CODE == "moss_transcribe_diarize":
        stream_builder = stream_builder.replace(
            "    def __init__(self, tokenizer: PreTrainedTokenizerBase, min_emit_interval_s: float) -> None:\n",
            "    def __init__(\n"
            "        self, tokenizer: PreTrainedTokenizerBase, min_emit_interval_s: float\n"
            "    ) -> None:\n",
        )
    else:
        pass
    request_source += stream_builder
    request_path.write_text(request_source)
    if MODEL_CODE == "whisper_asr":
        configuration = model / "config.py"
        configuration.write_text(replace_once(
            configuration.read_text(),
            "    enable_encoder_cuda_graph: bool | None = None\n",
            "    enable_encoder_cuda_graph: bool | None = None\n"
            "    stream_emit_interval_s: float | None = Field(default=None, ge=0)\n",
        ))
        stage = model / "stages.py"
        stage_source = replace_once(
            stage.read_text(), "    max_new_tokens: int = 256,\n",
            "    max_new_tokens: int = 256,\n    stream_emit_interval_s: float = 0.05,\n",
        )
        stage.write_text(replace_once(
            stage_source, "        max_new_tokens=max_new_tokens,\n",
            "        max_new_tokens=max_new_tokens,\n"
            "        stream_emit_interval_s=stream_emit_interval_s,\n",
        ))
        builder = model / "engine_builder.py"
        builder_source = replace_once(
            builder.read_text(), "    MAX_PREV_CONTEXT_TOKENS,\n",
            "    MAX_PREV_CONTEXT_TOKENS,\n    ExactASRStreamOutputBuilder,\n",
        )
        builder_source = replace_once(
            builder_source, "        max_new_tokens: int,\n",
            "        max_new_tokens: int,\n        stream_emit_interval_s: float,\n",
        )
        builder_source = replace_once(
            builder_source, "        self.max_new_tokens = max_new_tokens\n",
            "        self.max_new_tokens = max_new_tokens\n"
            "        self.stream_emit_interval_s = stream_emit_interval_s\n",
        )
        builder.write_text(replace_once(
            builder_source,
            "    def extra_scheduler_kwargs(self) -> SchedulerExtras[WhisperASRRequestData]:\n        return {\n",
            "    def extra_scheduler_kwargs(self) -> SchedulerExtras[WhisperASRRequestData]:\n        return {\n"
            "            \"stream_output_builder\": ExactASRStreamOutputBuilder(\n"
            "                self.tokenizer, self.stream_emit_interval_s\n            ),\n",
        ))
    else:
        replace_bodies(request_path, {
            "make_moss_transcribe_diarize_stream_output_builder":
                "return ExactASRStreamOutputBuilder(tokenizer, min_emit_interval_s)"
        })


if __name__ == "__main__":
    adapt_source(Path(sys.argv[1]))
