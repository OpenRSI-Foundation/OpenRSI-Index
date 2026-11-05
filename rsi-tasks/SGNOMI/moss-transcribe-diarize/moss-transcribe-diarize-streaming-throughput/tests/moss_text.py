"""Judge-owned plain-text extraction from MOSS-Transcribe-Diarize output.

Quality is scored exactly as in the other D257 ASR tasks: corpus WER on the
transcript. MOSS output additionally carries native timestamp and speaker
markers, which are removed here before normalization. Extraction never rejects
output; timestamp, speaker and segment structure are not part of the contract.
"""

import re


SPECIAL_TOKENS = re.compile(r"<\|(?:im_start|im_end|endoftext)\|>")
# Native timestamp ([12.34]) and speaker ([S01]) markers. Ordinary bracketed
# words such as [laughter] remain speech.
MARKERS = re.compile(r"\[(?:\d+(?:\.\d+)?|S\d{2,})\]")


def speech_text(raw_text: str, duration_seconds: float | None = None) -> str:
    """Return the spoken text in generated order with native markers removed.

    duration_seconds is accepted for call compatibility and is not validated.
    """
    del duration_seconds
    text = MARKERS.sub(" ", SPECIAL_TOKENS.sub(" ", raw_text))
    return " ".join(text.split())
