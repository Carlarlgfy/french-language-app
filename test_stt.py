#!/usr/bin/env python3
"""Transcribe test_recording.wav with faster-whisper (CPU, int8)."""
from faster_whisper import WhisperModel
import os

WAV_FILE = "test_recording.wav"
if not os.path.exists(WAV_FILE):
    print(f"ERROR: {WAV_FILE} not found. Run test_mic.py first.")
    exit(1)

print("Loading Whisper base.en model (downloads ~74MB on first run)...")
model = WhisperModel("base.en", device="cpu", compute_type="int8")

print(f"Transcribing {WAV_FILE}...")
segments, info = model.transcribe(WAV_FILE, beam_size=5, language="en")

text = " ".join(s.text.strip() for s in segments)
print(f"\nTranscription: {text!r}")
print(f"Language detected: {info.language} (probability {info.language_probability:.2f})")
