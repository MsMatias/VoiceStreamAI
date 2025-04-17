import os

import torch
from transformers import pipeline

from src.utils.audio_utils import convert_audio_bytes_to_numpy

from .asr_interface import ASRInterface


class WhisperASR(ASRInterface):
    def __init__(self, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = kwargs.get("model_name", "openai/whisper-large-v3")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device,
        )

    async def transcribe(self, client):
        audio_np = convert_audio_bytes_to_numpy(client.scratch_buffer)

        if client.config["language"] is not None:
            to_return = self.asr_pipeline(
                audio_np,
                generate_kwargs={"language": client.config["language"]},
            )["text"]
        else:
            to_return = self.asr_pipeline(audio_np)["text"]

        to_return = {
            "language": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
            "language_probability": None,
            "text": to_return.strip(),
            "words": "UNSUPPORTED_BY_HUGGINGFACE_WHISPER",
        }
        return to_return
