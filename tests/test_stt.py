import pytest
import numpy as np
from stt.audio_capture import AudioCapture

def test_audio_capture_init():
    cap = AudioCapture(sample_rate=16000, frame_duration_ms=30)
    assert cap.sample_rate == 16000
    assert cap.frame_duration_ms == 30

def test_vad_detects_silence():
    cap = AudioCapture()
    silent_frame = np.zeros(480, dtype=np.int16).tobytes()
    result = cap.is_speech(silent_frame)
    assert isinstance(result, bool)

def test_frame_size_calculation():
    cap = AudioCapture(sample_rate=16000, frame_duration_ms=30)
    # 16000 * 0.030 = 480 samples * 2 bytes = 960 bytes
    assert cap.frame_size_bytes == 960

def test_hotword_detector_init_no_key():
    from stt.hotword import HotwordDetector
    detector = HotwordDetector(access_key="", keyword="jarvis")
    assert detector.keyword == "jarvis"

def test_transcriber_init():
    from stt.transcriber import Transcriber
    t = Transcriber(model_size="tiny", device="cpu", compute_type="int8")
    assert t.model_size == "tiny"
    assert not t._loaded

@pytest.mark.asyncio
async def test_transcriber_transcribes_audio():
    """Uses tiny model on CPU — downloads on first run (~75MB)."""
    from stt.transcriber import Transcriber
    import struct, math
    sample_rate = 16000
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
               for i in range(sample_rate)]
    pcm = struct.pack(f"{len(samples)}h", *samples)
    t = Transcriber(model_size="tiny", device="cpu", compute_type="int8")
    result = await t.transcribe(pcm, sample_rate=sample_rate)
    assert isinstance(result, str)
