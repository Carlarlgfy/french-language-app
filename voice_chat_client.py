#!/usr/bin/env python3
"""
Voice Chat AI — French Tutor
Push-to-talk GUI. Bilingual EN/FR with auto language detection.
Voices: en_US-ryan-high / fr_FR-siwis-medium (Piper)
LLM: qwen3-8b on thebrain via LM Studio
"""
import os, sys, wave, tempfile, threading, time, subprocess
import tkinter as tk
from tkinter import scrolledtext
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from faster_whisper import WhisperModel
from piper.voice import PiperVoice

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR   = os.path.join(BASE_DIR, "voices")
THEBRAIN_URL = "http://192.168.2.12:1234/v1"
MODEL        = "qwen3-8b"
SAMPLE_RATE  = 16000

VOICE_FILES = {
    "en": ("en_US-ljspeech-high.onnx",    "en_US-ljspeech-high.onnx.json"),
    "fr": ("fr_FR-siwis-medium.onnx", "fr_FR-siwis-medium.onnx.json"),
}

SYSTEM_PROMPT = (
    "You are a warm, encouraging bilingual French-English conversation tutor. "
    "Rules you must always follow:\n"
    "- If the user speaks French, reply entirely in French.\n"
    "- If the user speaks English, reply in English.\n"
    "- Keep every reply to 2-4 sentences maximum — you are speaking aloud.\n"
    "- Never use markdown, bullet points, headers, asterisks, or any formatting.\n"
    "- Plain conversational sentences only.\n"
    "- When the user makes a French grammar mistake, correct it naturally inside "
    "your reply without making it feel like a lesson.\n"
    "- Be warm, natural, and encouraging.\n"
    "/no_think"
)

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "bg":       "#0e0b1f",
    "panel":    "#13102a",
    "border":   "#2b2060",
    "text":     "#ddd0ff",
    "dim":      "#7060a0",
    "you_en":   "#55ccff",
    "you_fr":   "#66ffaa",
    "ai_en":    "#bb99ff",
    "ai_fr":    "#ffcc55",
    "sys":      "#606080",
    "btn_idle": "#1e1a40",
    "btn_rec":  "#550020",
    "btn_proc": "#3a2a00",
    "btn_talk": "#003322",
    "ring_idle":"#6644cc",
    "ring_rec": "#ff2255",
    "ring_proc":"#ffaa00",
    "ring_talk":"#33cc77",
}

LANG_FLAG = {"en": "🇺🇸 English", "fr": "🇫🇷 Français"}

# ── App ───────────────────────────────────────────────────────────────────────
class VoiceChatApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.state = "loading"     # loading | idle | recording | processing | speaking
        self.audio_chunks: list    = []
        self.stream                = None
        self.voices: dict          = {}
        self.whisper               = None
        self.llm                   = None
        self.conversation: list    = []
        self.last_lang             = "en"

        self._build_ui()
        threading.Thread(target=self._load_models, daemon=True).start()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        r = self.root
        r.title("Voice Chat AI — French Tutor")
        r.configure(bg=C["bg"])
        r.geometry("720x620")
        r.minsize(560, 480)

        # Header
        hdr = tk.Frame(r, bg=C["bg"])
        hdr.pack(fill=tk.X, padx=18, pady=(14, 0))
        tk.Label(hdr, text="VOICE CHAT AI", bg=C["bg"], fg=C["text"],
                 font=("DejaVu Sans", 13, "bold")).pack(side=tk.LEFT)
        self.lang_lbl = tk.Label(hdr, text="", bg=C["bg"], fg=C["dim"],
                                  font=("DejaVu Sans", 10))
        self.lang_lbl.pack(side=tk.RIGHT)

        # Transcript
        tf = tk.Frame(r, bg=C["panel"],
                      highlightbackground=C["border"], highlightthickness=1)
        tf.pack(fill=tk.BOTH, expand=True, padx=18, pady=10)

        self.log = scrolledtext.ScrolledText(
            tf, bg=C["panel"], fg=C["text"],
            font=("DejaVu Sans", 12), wrap=tk.WORD,
            relief=tk.FLAT, padx=14, pady=10,
            state=tk.DISABLED, cursor="arrow",
        )
        self.log.pack(fill=tk.BOTH, expand=True)

        for tag, fg, bold in [
            ("you_en", C["you_en"], True),
            ("you_fr", C["you_fr"], True),
            ("ai_en",  C["ai_en"],  True),
            ("ai_fr",  C["ai_fr"],  True),
            ("sys",    C["sys"],    False),
            ("body",   C["text"],   False),
        ]:
            font = ("DejaVu Sans", 12, "bold") if bold else ("DejaVu Sans", 12)
            self.log.tag_config(tag, foreground=fg, font=font)

        # Bottom bar
        bar = tk.Frame(r, bg=C["bg"])
        bar.pack(fill=tk.X, padx=18, pady=(0, 16))

        self.status_var = tk.StringVar(value="Starting up…")
        tk.Label(bar, textvariable=self.status_var, bg=C["bg"], fg=C["dim"],
                 font=("DejaVu Sans", 10)).pack()

        # PTT button canvas
        self.canvas = tk.Canvas(bar, width=130, height=130,
                                bg=C["bg"], highlightthickness=0)
        self.canvas.pack(pady=(8, 4))
        self.canvas.bind("<Button-1>", self._click)

        self.hint_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self.hint_var, bg=C["bg"], fg=C["dim"],
                 font=("DejaVu Sans", 9)).pack()

        # Audio test button
        test_btn = tk.Button(bar, text="⬤ HOLD TO TEST AUDIO", bg="#1a0030", fg="#aa66ff",
                             font=("DejaVu Sans", 9, "bold"), relief=tk.FLAT,
                             activebackground="#330055", activeforeground="#ffffff",
                             bd=0, padx=10, pady=4)
        test_btn.pack(pady=(10, 0))
        test_btn.bind("<ButtonPress-1>",   self._beep_start)
        test_btn.bind("<ButtonRelease-1>", self._beep_stop)

        self._draw_btn("loading")

    def _draw_btn(self, state: str):
        cfg = {
            "loading":    (C["btn_idle"], C["dim"],       "…",  "loading models"),
            "idle":       (C["btn_idle"], C["ring_idle"], "🎙", "click to speak"),
            "recording":  (C["btn_rec"],  C["ring_rec"],  "⏹", "click to send"),
            "processing": (C["btn_proc"], C["ring_proc"], "⋯",  "processing…"),
            "speaking":   (C["btn_talk"], C["ring_talk"], "♪",  "speaking…"),
        }
        bg, ring, symbol, hint = cfg.get(state, cfg["idle"])
        self.hint_var.set(hint)
        c = self.canvas
        c.delete("all")
        # Glow rings
        for i in range(10, 0, -2):
            c.create_oval(25 - i, 25 - i, 105 + i, 105 + i,
                          outline=ring, width=1)
        # Main circle
        c.create_oval(25, 25, 105, 105, fill=bg, outline=ring, width=2)
        # Icon
        c.create_text(65, 65, text=symbol, fill=ring,
                      font=("DejaVu Sans", 30))

    def _set_state(self, state: str):
        self.state = state
        self.root.after(0, lambda s=state: self._draw_btn(s))

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_models(self):
        try:
            self._status("Loading Whisper (base multilingual)…")
            self.whisper = WhisperModel("base", device="cpu", compute_type="int8")

            for lang, (onnx, jsn) in VOICE_FILES.items():
                self._status(f"Loading {LANG_FLAG[lang]} voice…")
                self.voices[lang] = PiperVoice.load(
                    os.path.join(VOICES_DIR, onnx),
                    config_path=os.path.join(VOICES_DIR, jsn),
                    use_cuda=False,
                )

            self._status("Connecting to LM Studio on thebrain…")
            self.llm = OpenAI(base_url=THEBRAIN_URL, api_key="lm-studio")
            self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Open always-on input stream (collects only when recording)
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=self._audio_cb,
            )
            self.stream.start()

            self._set_state("idle")
            self._status("Ready — click the mic to speak")
            self.root.after(0, lambda: self.lang_lbl.config(text="speak EN or FR"))
            self._log("AI", "en",
                      "Bonjour ! I'm your French tutor. Speak in English or French "
                      "— I'll match your language automatically.", is_ai=True)
        except Exception as e:
            self._status(f"Startup error: {e}")

    # ── Audio ─────────────────────────────────────────────────────────────────
    def _audio_cb(self, indata, frames, time, status):
        if self.state == "recording":
            self.audio_chunks.append(indata.copy())

    def _click(self, _event=None):
        if self.state == "idle":
            self._start_rec()
        elif self.state == "recording":
            self._stop_rec()

    def _start_rec(self):
        self.audio_chunks = []
        self._set_state("recording")
        self._status("Listening… click again when done")

    def _stop_rec(self):
        chunks = self.audio_chunks[:]      # snapshot before state change
        self._set_state("processing")
        self._status("Transcribing…")
        threading.Thread(target=self._process, args=(chunks,), daemon=True).start()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    def _process(self, chunks: list):
        wav_path = None
        try:
            if not chunks:
                self._status("No audio recorded — try again")
                time.sleep(1.5)
                self._set_state("idle")
                self._status("Ready — click the mic to speak")
                return

            # Save to temp WAV
            audio = np.concatenate(chunks, axis=0)
            if audio.shape[0] < SAMPLE_RATE * 0.4:   # under 0.4 s — too short
                self._status("Too short — hold longer next time")
                time.sleep(1.5)
                self._set_state("idle")
                self._status("Ready — click the mic to speak")
                return

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            sf.write(wav_path, audio, SAMPLE_RATE)

            # Transcribe + detect language
            segments, info = self.whisper.transcribe(
                wav_path, beam_size=5, language=None   # auto-detect
            )
            user_text = " ".join(s.text.strip() for s in segments).strip()
            detected = info.language if info.language in ("en", "fr") else "en"
            self.last_lang = detected

            if not user_text:
                self._status("No speech detected — try again")
                time.sleep(1.5)
                self._set_state("idle")
                self._status("Ready — click the mic to speak")
                return

            # Show user line + update language badge
            self.root.after(0, lambda: self._log("You", detected, user_text, is_ai=False))
            flag = LANG_FLAG[detected]
            self.root.after(0, lambda: self.lang_lbl.config(text=f"Detected: {flag}"))

            # Quit phrase check
            quit_words = {"quit", "exit", "quitter", "arrêter", "stop"}
            if any(w in user_text.lower() for w in quit_words):
                self._speak("Au revoir !", "fr")
                self.root.after(2500, self.root.destroy)
                return

            # LLM — include a language directive so the model stays consistent
            lang_directive = "(Répondre en français.)" if detected == "fr" \
                             else "(Reply in English.)"
            self.conversation.append({
                "role": "user",
                "content": f"{user_text}  {lang_directive}",
            })

            self._status("Thinking on thebrain…")
            resp = self.llm.chat.completions.create(
                model=MODEL,
                messages=self.conversation,
                max_tokens=300,
            )
            reply = resp.choices[0].message.content.strip()

            # Fallback: if model produced only reasoning content (Qwen3 thinking mode)
            if not reply and hasattr(resp.choices[0].message, "reasoning_content"):
                reply = resp.choices[0].message.reasoning_content.strip()

            if not reply:
                reply = "Je ne sais pas." if detected == "fr" else "I'm not sure."

            self.conversation.append({"role": "assistant", "content": reply})

            # Show AI line
            self.root.after(0, lambda: self._log("AI", detected, reply, is_ai=True))

            # Speak
            self._set_state("speaking")
            self._status(f"Speaking in {flag}…")
            self._speak(reply, detected)

        except Exception as e:
            self._status(f"Error: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            self._set_state("idle")
            self._status("Ready — click the mic to speak")

    def _speak(self, text: str, lang: str):
        voice = self.voices.get(lang) or self.voices.get("en")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav_path = f.name
        try:
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(voice.config.sample_rate)
                voice.synthesize(text, wf)

            if self.stream:
                self.stream.stop()
            try:
                audio, sr = sf.read(wav_path, dtype="float32")
                # Device 15 = PipeWire — handles sample rate conversion and
                # routes to whatever output the system is configured for
                sd.play(audio, sr, device=15)
                sd.wait()
            except Exception as play_err:
                self._log_err(f"sd.play pw failed: {play_err}")
                try:
                    # Fallback: pulse device
                    sd.play(audio, sr, device=16)
                    sd.wait()
                except Exception as e2:
                    self._log_err(f"sd.play pulse failed: {e2}")
                    self._status(f"Audio error: {e2}")
            finally:
                if self.stream:
                    self.stream.start()
        except Exception as e:
            self._log_err(f"_speak error: {e}")
            self._status(f"TTS error: {e}")
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def _log_err(self, msg: str):
        with open("/tmp/voicechat.log", "a") as f:
            f.write(f"{msg}\n")

    def _beep_start(self, _=None):
        self._beeping = True
        threading.Thread(target=self._play_beep, daemon=True).start()

    def _beep_stop(self, _=None):
        self._beeping = False
        sd.stop()

    def _play_beep(self):
        sr = 48000
        freq = 440
        t = np.linspace(0, 1, sr, endpoint=False)
        tone = (np.sin(2 * np.pi * freq * t) * 0.3).astype("float32")
        try:
            sd.play(np.tile(tone, 10), sr, device=15)
            sd.wait()
        except Exception as e:
            self._log_err(f"beep device 15 failed: {e}")
            try:
                sd.play(np.tile(tone, 10), sr, device=16)
                sd.wait()
            except Exception as e2:
                self._log_err(f"beep device 16 failed: {e2}")

    # ── Transcript ────────────────────────────────────────────────────────────
    def _log(self, speaker: str, lang: str, text: str, is_ai: bool):
        t = self.log
        t.configure(state=tk.NORMAL)
        tag = f"{'ai' if is_ai else 'you'}_{lang}"
        label = f"{'AI' if is_ai else 'You'} [{lang.upper()}]"
        t.insert(tk.END, f"\n{label}\n", tag)
        t.insert(tk.END, f"{text}\n", "body")
        t.configure(state=tk.DISABLED)
        t.see(tk.END)

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _status(self, msg: str):
        self.root.after(0, lambda: self.status_var.set(msg))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    root = tk.Tk()
    icon = os.path.join(BASE_DIR, "voicechat_icon.png")
    if os.path.exists(icon):
        try:
            root.iconphoto(True, tk.PhotoImage(file=icon))
        except Exception:
            pass
    VoiceChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
