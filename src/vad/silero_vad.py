from math import floor

from silero_vad import load_silero_vad, get_speech_timestamps

from src.utils.audio_utils import convert_audio_bytes_to_numpy
from src.utils.base_logger import logger

from .vad_interface import VADInterface

class SileroVAD(VADInterface):
    """
    Pyannote-based implementation of the VADInterface that works with in-memory audio.
    """

    def __init__(self, **kwargs):
        """
        Initializes RVADFast's VAD pipeline.
        """
        self.model = load_silero_vad()

    async def detect_activity(self, client):
        # Convert bytearray to numpy array
        audio_np = convert_audio_bytes_to_numpy(client.scratch_buffer)
        
        speech_timestamps = get_speech_timestamps(audio_np, self.model)
        
        # It returns ms
        new_timestamps = [
            {'start': floor(timestamp['start'] / client.sampling_rate), 'end': floor(timestamp['end'] / client.sampling_rate)}
            for timestamp in speech_timestamps
        ]
        
        return new_timestamps