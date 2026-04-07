import pytest
import asyncio
from tts.speaker import Speaker

def test_speaker_detects_language_en():
    sp = Speaker(voice_en="en-US-JennyNeural", voice_cs="cs-CZ-VlastaNeural")
    assert sp._detect_language("Hello, how are you?") == "en"

def test_speaker_detects_language_cs():
    sp = Speaker(voice_en="en-US-JennyNeural", voice_cs="cs-CZ-VlastaNeural")
    assert sp._detect_language("Ahoj, jak se máš?") == "cs"

def test_speaker_detects_language_cs_short():
    sp = Speaker()
    # needs >= 2 Czech chars
    assert sp._detect_language("Dobrý den, jak se máte?") == "cs"

def test_speaker_detects_language_en_no_cs_chars():
    sp = Speaker()
    assert sp._detect_language("What is the weather today?") == "en"

@pytest.mark.asyncio
async def test_speaker_can_be_interrupted():
    sp = Speaker(voice_en="en-US-JennyNeural", voice_cs="cs-CZ-VlastaNeural")
    sp.interrupt()  # Should not raise even when not speaking
    assert not sp.is_speaking

@pytest.mark.asyncio
async def test_speaker_interrupt_thread_safe():
    """interrupt() with a running loop should use call_soon_threadsafe."""
    sp = Speaker()
    loop = asyncio.get_event_loop()
    sp.interrupt(loop=loop)
    # call_soon_threadsafe schedules the callback; yield to let the loop run it
    await asyncio.sleep(0)
    assert sp._cancel_event.is_set()
