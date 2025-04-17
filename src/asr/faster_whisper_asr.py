import os
import torch
from faster_whisper import WhisperModel

from src.utils.audio_utils import convert_audio_bytes_to_numpy

from .asr_interface import ASRInterface

class FasterWhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        model_size = kwargs.get("model_size", "tiny")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        self.asr_pipeline = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )

    async def transcribe(self, client):
        audio_np = convert_audio_bytes_to_numpy(client.scratch_buffer)

        language = client.config["language"].lower()
        segments, info = self.asr_pipeline.transcribe(
            audio_np, language=language
        )

        segments = list(segments)  # The transcription will actually run here.

        flattened_words = [
            word for segment in segments for word in segment.words
        ]

        to_return = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": " ".join([s.text.strip() for s in segments]),
            "words": [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                }
                for w in flattened_words
            ],
        }
        return to_return
