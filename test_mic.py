#!/usr/bin/env python3
"""Record 5 seconds from the default mic, save to test_recording.wav, play it back."""
import sounddevice as sd
import soundfile as sf
import numpy as np

SAMPLE_RATE = 16000
DURATION = 5
OUT_FILE = "test_recording.wav"

print("Available audio devices:")
print(sd.query_devices())
print()

print(f"Recording {DURATION} seconds from default mic... speak now!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
sd.wait()
sf.write(OUT_FILE, audio, SAMPLE_RATE)
print(f"Saved to {OUT_FILE}")

print("Playing back recording...")
data, sr = sf.read(OUT_FILE)
sd.play(data, sr)
sd.wait()
print("Done. If you heard yourself, the mic is working.")
