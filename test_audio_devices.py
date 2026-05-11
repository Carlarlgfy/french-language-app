#!/usr/bin/env python3
"""Test audio devices and find working output"""
import sounddevice as sd
import numpy as np

print("Available Audio Devices:")
print(sd.query_devices())

print("\n" + "="*60)
print("Testing default output device...")
print("="*60)

# Generate test tone
sr = 22050
freq = 440
duration = 1
t = np.linspace(0, duration, int(sr * duration))
tone = (np.sin(2 * np.pi * freq * t) * 0.1).astype('float32')

try:
    print("Attempting to play on default device...")
    sd.play(tone, sr)
    sd.wait()
    print("✓ Success! Audio played on default device.")
except Exception as e:
    print(f"✗ Failed: {e}")

print("\nDefault output device ID:", sd.default.device[1])
