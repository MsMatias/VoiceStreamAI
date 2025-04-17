import os
import io
import soundfile as sf
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

from src.utils.audio_utils import convert_audio_bytes_to_numpy

from .vad_interface import VADInterface


class PyannoteVAD(VADInterface):
    """
    Pyannote-based implementation of the VADInterface.
    """

    def __init__(self, **kwargs):
        """
        Initializes Pyannote's VAD pipeline.

        Args:
            model_name (str): The model name for Pyannote.
            auth_token (str, optional): Authentication token for Hugging Face.
        """

        model_name = kwargs.get("model_name", "pyannote/segmentation")

        auth_token = os.environ.get("PYANNOTE_AUTH_TOKEN")
        if not auth_token:
            auth_token = kwargs.get("auth_token")

        if auth_token is None:
            raise ValueError(
                "Missing required env var in PYANNOTE_AUTH_TOKEN or argument "
                "in --vad-args: 'auth_token'"
            )

        pyannote_args = kwargs.get(
            "pyannote_args",
            {
                "onset": 0.5,
                "offset": 0.5,
                "min_duration_on": 0.3,
                "min_duration_off": 0.3,
            },
        )
        self.model = Model.from_pretrained(
            model_name, use_auth_token=auth_token
        )
        self.vad_pipeline = VoiceActivityDetection(segmentation=self.model)
        self.vad_pipeline.instantiate(pyannote_args)

    async def detect_activity(self, client):
        audio_np = convert_audio_bytes_to_numpy(client.scratch_buffer)
        
        # Create an in-memory audio file
        audio_buffer = io.BytesIO()
        # Save as WAV at 16kHz sample rate
        sf.write(audio_buffer, audio_np, 16000, format='WAV')
        
        # Reset buffer position for reading
        audio_buffer.seek(0)
        
        # Process with Pyannote directly from the in-memory buffer
        vad_results = self.vad_pipeline(audio_buffer)

        vad_segments = []
        if len(vad_results) > 0:
            vad_segments = [
                {"start": segment.start, "end": segment.end, "confidence": 1.0}
                for segment in vad_results.itersegments()
            ]
        return vad_segments
