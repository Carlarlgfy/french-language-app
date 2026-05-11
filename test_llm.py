#!/usr/bin/env python3
"""Test connection to LM Studio on thebrain."""
from openai import OpenAI

THEBRAIN_URL = "http://192.168.2.12:1234/v1"
MODEL = "qwen3-8b"

client = OpenAI(base_url=THEBRAIN_URL, api_key="lm-studio")

print(f"Connecting to LM Studio at {THEBRAIN_URL}...")
models = client.models.list()
print(f"Available models: {[m.id for m in models.data]}")

print(f"\nSending test message to {MODEL}...")
response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful voice assistant. Keep answers short and conversational. /no_think"},
        {"role": "user", "content": "Say hello to Carl in one sentence."},
    ],
    max_tokens=100,
)
reply = response.choices[0].message.content.strip()
print(f"\nModel reply: {reply}")
