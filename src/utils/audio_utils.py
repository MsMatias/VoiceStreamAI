from numpy import frombuffer, int16, float32

def convert_audio_bytes_to_numpy(audio_bytes):
    """
    Convert raw audio bytes from scratch_buffer (bytearray) 
    directly to the numpy format required by Whisper and VAD
    
    :param audio_bytes: Raw audio bytes as bytearray
    :return: Numpy array with the audio data in the format expected by Whisper
    """
    # Convert bytearray directly to numpy array
    # Assuming 16-bit PCM audio format
    audio_as_np_int16 = frombuffer(audio_bytes, dtype=int16)
    
    # Convert to float32 and normalize to [-1, 1] range as expected by Whisper
    audio_as_np_float32 = audio_as_np_int16.astype(float32) / 32768.0
    
    return audio_as_np_float32