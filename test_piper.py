#!/usr/bin/env python3
"""Test Piper TTS directly"""
import os, wave, tempfile
from piper.voice import PiperVoice

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR = os.path.join(BASE_DIR, "voices")

def test_voice(lang, text):
    print(f"\n=== Testing {lang.upper()} voice ===")
    voice_file = f"{'en_US-ljspeech-high' if lang == 'en' else 'fr_FR-siwis-medium'}.onnx"
    config_file = voice_file + ".json"

    voice_path = os.path.join(VOICES_DIR, voice_file)
    config_path = os.path.join(VOICES_DIR, config_file)

    print(f"Voice file: {voice_path}")
    print(f"Config file: {config_path}")
    print(f"Text: {text}")

    try:
        voice = PiperVoice.load(voice_path, config_path=config_path, use_cuda=False)
        print("Voice loaded successfully")

        wav_path = f"/tmp/test_{lang}.wav"
        with wave.open(wav_path, "wb") as wf:
            voice.synthesize_wav(text, wf)

        file_size = os.path.getsize(wav_path)
        print(f"Audio generated: {file_size} bytes at {wav_path}")

        import subprocess
        print("Playing audio...")
        result = subprocess.run(["paplay", wav_path], capture_output=True, timeout=10)
        if result.returncode == 0:
            print("Audio played successfully")
        else:
            print(f"Audio play failed: {result.stderr.decode()}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_voice("en", "Hello, this is a test of the English voice.")
    test_voice("fr", "Bonjour, ceci est un test de la voix française.")
