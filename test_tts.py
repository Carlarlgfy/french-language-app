#!/usr/bin/env python3
"""Test Piper TTS — speaks a test phrase through charles speakers."""
import wave
import subprocess
import os

VOICE_ONNX = os.path.join(os.path.dirname(__file__), "voices", "en_US-lessac-medium.onnx")
VOICE_JSON = VOICE_ONNX + ".json"
TEST_TEXT = "Hello Carl, the voice output is working on charles."
OUT_WAV = "/tmp/tts_test.wav"

from piper.voice import PiperVoice

print(f"Loading Piper voice...")
voice = PiperVoice.load(VOICE_ONNX, config_path=VOICE_JSON, use_cuda=False)

print(f"Synthesizing: {TEST_TEXT!r}")
with wave.open(OUT_WAV, "wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(voice.config.sample_rate)
    voice.synthesize(TEST_TEXT, wav_file)

print(f"Playing with aplay...")
result = subprocess.run(["aplay", OUT_WAV], capture_output=True, text=True)
if result.returncode == 0:
    print("SUCCESS: You should have heard the phrase.")
else:
    print(f"aplay error: {result.stderr}")
    print("Try: paplay /tmp/tts_test.wav")
