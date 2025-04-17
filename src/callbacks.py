from typing import Callable, Any, Optional, Awaitable

class AudioProcessingCallbacks:
    """
    Callback interface for audio processing events.
    
    This class defines callback functions that can be registered to handle
    different events that occur during audio processing, such as:
    - Transcription completion
    
    Callbacks are defined as async functions to support asynchronous operations.
    """
    
    def __init__(
        self,
        on_transcription_complete: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """
        Initialize the callback interface.
        
        Args:
            on_transcription_complete: Called when transcription is complete with the
                                      transcribed text.
        """
        self.on_transcription_complete = on_transcription_complete
        
    async def trigger_transcription_complete(self, text: str):
        """
        Trigger the transcription complete callback.
        
        Args:
            text: The transcribed text.
        """
        if self.on_transcription_complete:
            await self.on_transcription_complete(text)