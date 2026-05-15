#!/usr/bin/env python3
"""
Voice Chat AI — French Tutor (Pygame Version)
Push-to-talk GUI. Bilingual EN/FR with auto language detection.
Voices: en_US-ryan-high / fr_FR-siwis-medium (Piper)
LLM: qwen3-8b on thebrain via LM Studio
"""
import os, sys, wave, tempfile, threading, time, math, subprocess, re, io, json, shlex
import urllib.request
import urllib.parse
import numpy as np
import pygame
import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from faster_whisper import WhisperModel
from piper.voice import PiperVoice, SynthesisConfig

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
VOICES_DIR   = os.path.join(BASE_DIR, "voices")
SAMPLE_RATE  = 16000
CONFIG_PATH  = os.path.join(BASE_DIR, "voicechat_config.json")
LOG_PATH     = "/tmp/voicechat.log"

DEFAULT_CONFIG = {
    "brain_url": "http://192.168.2.12:1234/v1",
    "model": "google/gemma-3-4b",
    "max_history_turns": 10,
    "brain_start_command": [],
    "brain_start_url": "",
    "brain_start_url_method": "POST",
}

def _load_config():
    config = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                config.update(loaded)
        except Exception as e:
            print(f"[CONFIG] Failed to read {CONFIG_PATH}: {e}", file=sys.stderr)

    if os.environ.get("VOICECHAT_BRAIN_URL"):
        config["brain_url"] = os.environ["VOICECHAT_BRAIN_URL"]
    if os.environ.get("VOICECHAT_MODEL"):
        config["model"] = os.environ["VOICECHAT_MODEL"]
    if os.environ.get("VOICECHAT_BRAIN_START_COMMAND"):
        config["brain_start_command"] = shlex.split(os.environ["VOICECHAT_BRAIN_START_COMMAND"])

    return config

CONFIG       = _load_config()
THEBRAIN_URL = CONFIG["brain_url"].rstrip("/")
MODEL        = CONFIG["model"]

VOICE_FILES = {
    "en": ("en_US-ljspeech-high.onnx",    "en_US-ljspeech-high.onnx.json"),
    "fr": ("fr_FR-siwis-medium.onnx", "fr_FR-siwis-medium.onnx.json"),
}

SYSTEM_PROMPT = (
    "You are a warm, encouraging bilingual French-English conversation tutor. "
    "Each user message ends with an instruction in [square brackets] — follow it exactly.\n"
    "General rules:\n"
    "- Keep every reply to 2-4 sentences maximum — you are speaking aloud.\n"
    "- Never use markdown, bullet points, headers, asterisks, or any formatting.\n"
    "- Plain conversational sentences only.\n"
    "- Never repeat or translate the user's question back to them. Just answer it.\n"
    "- When the user makes a French grammar mistake, correct it naturally.\n"
    "- Be warm, natural, and encouraging.\n"
    "Language tagging rule — always apply this:\n"
    "Wrap every run of French words with [FR]...[/FR] and every run of English words "
    "with [EN]...[/EN]. Apply tags to ALL words in your reply, even in single-language "
    "replies. Example of a mixed reply: "
    "[EN]The word[/EN] [FR]alors[/FR] [EN]means 'then' or 'so'.[/EN] "
    "Example of a French-only reply: [FR]Bonjour ! Comment puis-je vous aider ?[/FR]"
)

# ── Language tag helpers ───────────────────────────────────────────────────────
_TAG_RE = re.compile(r'\[(FR|EN)\](.*?)\[/\1\]', re.IGNORECASE | re.DOTALL)

def _parse_lang_tags(text: str, default_lang: str = "en") -> list:
    """Return [(lang, text), ...] from a tagged LLM response.
    Untagged runs fall back to default_lang."""
    segments = []
    last_end = 0
    for m in _TAG_RE.finditer(text):
        before = text[last_end:m.start()].strip()
        if before:
            segments.append((default_lang, before))
        seg_text = m.group(2).strip()
        if seg_text:
            segments.append((m.group(1).lower(), seg_text))
        last_end = m.end()
    after = text[last_end:].strip()
    if after:
        segments.append((default_lang, after))
    return segments or [(default_lang, text)]

def _strip_lang_tags(text: str) -> str:
    """Remove [FR]/[EN] tags, keeping the inner text."""
    return _TAG_RE.sub(lambda m: m.group(2), text).strip()

# Patterns that signal a cross-language translation question
_CROSS_EN = re.compile(
    r"\b(how (do you |to )?say|what does .+ mean|translate|in french|en fran[cç]ais"
    r"|what('?s| is) .+ in french)\b",
    re.IGNORECASE,
)
_CROSS_FR = re.compile(
    r"\b(comment dit-?on|que (veut dire|signifie)|qu'?est-?ce que .+ veut dire"
    r"|en anglais|comment (se dit|s'?appelle|traduit))\b",
    re.IGNORECASE,
)

def _build_lang_directive(text: str, detected: str) -> str:
    is_cross = bool(_CROSS_EN.search(text)) or bool(_CROSS_FR.search(text))
    if is_cross:
        if detected == "fr":
            return (
                "Cross-language question in French. Answer in French but state the "
                "English word/phrase. Tag ALL words with [FR]...[/FR] or [EN]...[/EN]."
            )
        else:
            return (
                "Cross-language question in English. Answer in English but state the "
                "French word/phrase. Tag ALL words with [FR]...[/FR] or [EN]...[/EN]."
            )
    if detected == "fr":
        return "Reply in French only. Tag every word: [FR]...[/FR]."
    return "Reply in English only. Tag every word: [EN]...[/EN]."

# ── Colours ───────────────────────────────────────────────────────────────────
C = {
    "bg":       (14, 11, 31),
    "panel":    (19, 16, 42),
    "border":   (43, 32, 96),
    "text":     (221, 208, 255),
    "dim":      (112, 96, 160),
    "you_en":   (85, 204, 255),
    "you_fr":   (102, 255, 170),
    "ai_en":    (187, 153, 255),
    "ai_fr":    (255, 204, 85),
    "sys":      (96, 96, 128),
    "btn_idle": (30, 26, 64),
    "btn_rec":  (85, 0, 32),
    "btn_proc": (58, 42, 0),
    "btn_talk": (0, 51, 34),
    "ring_idle":(102, 68, 204),
    "ring_rec": (255, 34, 85),
    "ring_proc":(255, 170, 0),
    "ring_talk":(51, 204, 119),
}

LANG_FLAG = {"en": "🇺🇸 English", "fr": "🇫🇷 Français"}

# ── App ───────────────────────────────────────────────────────────────────────
class VoiceChatApp:
    def __init__(self):
        pygame.init()
        self.width = 720
        self.height = 620
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Voice Chat AI — French Tutor")
        
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.Font(None, 28)
        self.font_normal = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 18)
        
        # State
        self.state = "loading"     # loading | idle | recording | processing | speaking
        self.audio_chunks = []
        self.stream = None
        self.voices = {}
        self.whisper = None
        self.llm = None
        self.conversation = []
        self.last_lang = "en"
        self.config = CONFIG
        self.brain_url = THEBRAIN_URL
        self.model = MODEL
        self.max_history_turns = max(1, int(self.config.get("max_history_turns", 10)))
        self.status_msg = "Starting up…"
        self.hint_msg = "loading models"
        self.lang_badge = self._brain_label()
        self.scroll_offset = 0
        self.max_scroll = 0
        
        # Word highlighting for speech
        self.speaking_line_idx = None  # Which transcript line is speaking
        self.speaking_word_idx = None  # Which word in that line
        
        # Transcript lines: (speaker, lang, text, is_ai)
        self.transcript = []
        
        # Buttons
        self.btn_mic_rect = pygame.Rect(self.width // 2 - 65, self.height - 200, 130, 130)
        self.start_btn_rect = pygame.Rect(self.width // 2 - 170, self.height - 60, 160, 40)
        self.test_btn_rect = pygame.Rect(self.width // 2 + 10, self.height - 60, 160, 40)
        
        # Test audio state
        self.beeping = False
        self.brain_starting = False

        # Speed slider: 0.5 (50%) to 200%
        self.speech_rate = 1.0
        self.slider_dragging = False
        self._slider_track = None

        # Scroll
        self._scroll_to_bottom = False

        # Brain connection state
        self.brain_connected = False

        # Load models in background, then start brain watchdog
        threading.Thread(target=self._load_models, daemon=True).start()
        threading.Thread(target=self._brain_watchdog, daemon=True).start()

    # ── Rendering ─────────────────────────────────────────────────────────────
    def render(self):
        self.screen.fill(C["bg"])
        
        # Header
        title = self.font_title.render("VOICE CHAT AI", True, C["text"])
        self.screen.blit(title, (18, 14))
        
        lang_surf = self.font_small.render(self.lang_badge, True, C["dim"])
        self.screen.blit(lang_surf, (self.width - lang_surf.get_width() - 18, 18))
        
        # Transcript area
        transcript_rect = pygame.Rect(18, 50, self.width - 36, self.height - 280)
        pygame.draw.rect(self.screen, C["panel"], transcript_rect)
        pygame.draw.rect(self.screen, C["border"], transcript_rect, 1)
        
        # Render transcript with scrolling and word highlighting
        clip = self.screen.get_clip()
        self.screen.set_clip(transcript_rect)
        
        y = transcript_rect.y + 10 - self.scroll_offset
        for line_idx, (speaker, lang, text, is_ai) in enumerate(self.transcript):
            if speaker == "SYS":
                tag_color = C["sys"]
                label = "●"
            else:
                tag_color = self._get_tag_color(is_ai, lang)
                label = f"{'AI' if is_ai else 'You'} [{lang.upper()}]"
            label_surf = self.font_normal.render(label, True, tag_color)
            self.screen.blit(label_surf, (transcript_rect.x + 14, y))
            y += label_surf.get_height() + 4
            
            # Word wrap with highlighting
            wrapped = self._wrap_text(text, transcript_rect.width - 28)
            words = text.split()
            word_idx = 0
            
            for line in wrapped:
                x = transcript_rect.x + 14
                line_words = line.split()
                
                for word in line_words:
                    # Check if this word should be highlighted
                    is_highlighted = (line_idx == self.speaking_line_idx and 
                                     word_idx == self.speaking_word_idx)
                    
                    if is_highlighted:
                        # Draw highlight background
                        word_surf = self.font_normal.render(word + " ", True, C["bg"])
                        highlight_rect = word_surf.get_rect(topleft=(x, y))
                        highlight_rect.inflate_ip(4, 2)
                        pygame.draw.rect(self.screen, C["text"], highlight_rect)
                        self.screen.blit(word_surf, (x, y))
                    else:
                        word_surf = self.font_normal.render(word + " ", True, C["text"])
                        self.screen.blit(word_surf, (x, y))
                    
                    x += word_surf.get_width()
                    word_idx += 1
                
                y += self.font_normal.get_height()
            
            y += 8
        
        # y + scroll_offset gives scroll-independent content bottom
        self.max_scroll = max(0, y + self.scroll_offset - transcript_rect.y - transcript_rect.height)
        self.screen.set_clip(clip)

        # Auto-scroll to bottom when new message logged
        if self._scroll_to_bottom:
            self.scroll_offset = self.max_scroll
            self._scroll_to_bottom = False

        # Clamp in case window was resized
        self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset))

        # Scrollbar
        if self.max_scroll > 0:
            sb_w = 6
            sb_x = transcript_rect.right - sb_w - 2
            sb_track_h = transcript_rect.height - 4
            sb_y = transcript_rect.y + 2
            thumb_h = max(30, int(sb_track_h * transcript_rect.height /
                                  (transcript_rect.height + self.max_scroll)))
            thumb_y = sb_y + int((sb_track_h - thumb_h) *
                                  self.scroll_offset / self.max_scroll)
            pygame.draw.rect(self.screen, C["border"],
                             (sb_x, sb_y, sb_w, sb_track_h), border_radius=3)
            pygame.draw.rect(self.screen, C["dim"],
                             (sb_x, thumb_y, sb_w, thumb_h), border_radius=3)
        
        # Status bar
        status_surf = self.font_small.render(self.status_msg, True, C["dim"])
        self.screen.blit(status_surf, (18, self.height - 240))

        # Speed slider
        self._draw_speed_slider()

        # Mic button
        self._draw_mic_button()
        
        # Hint
        hint_surf = self.font_small.render(self.hint_msg, True, C["dim"])
        self.screen.blit(hint_surf, (self.width // 2 - hint_surf.get_width() // 2, 
                                     self.height - 90))
        
        # Start brain button
        start_bg = (20, 36, 48) if not self.brain_connected else (14, 44, 30)
        start_fg = (85, 204, 255) if not self.brain_connected else C["ring_talk"]
        start_label = "BRAIN ONLINE" if self.brain_connected else "START BRAIN"
        if self.brain_starting:
            start_label = "STARTING..."
        pygame.draw.rect(self.screen, start_bg, self.start_btn_rect)
        pygame.draw.rect(self.screen, start_fg, self.start_btn_rect, 1)
        start_txt = self.font_small.render(start_label, True, start_fg)
        self.screen.blit(start_txt, (self.start_btn_rect.centerx - start_txt.get_width() // 2,
                                     self.start_btn_rect.y + 12))

        # Test audio button
        pygame.draw.rect(self.screen, (26, 0, 48), self.test_btn_rect)
        pygame.draw.rect(self.screen, (170, 102, 255), self.test_btn_rect, 1)
        test_txt = self.font_small.render("HOLD TEST AUDIO", True, (170, 102, 255))
        self.screen.blit(test_txt, (self.test_btn_rect.centerx - test_txt.get_width() // 2,
                                    self.test_btn_rect.y + 12))
        
        pygame.display.flip()

    def _get_tag_color(self, is_ai, lang):
        if is_ai:
            return C["ai_fr"] if lang == "fr" else C["ai_en"]
        else:
            return C["you_fr"] if lang == "fr" else C["you_en"]

    def _wrap_text(self, text, width):
        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test = current_line + word + " "
            if self.font_normal.size(test)[0] < width:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = word + " "
        if current_line:
            lines.append(current_line.strip())
        return lines

    def _brain_label(self):
        parsed = urllib.parse.urlparse(self.brain_url)
        host = parsed.netloc or self.brain_url
        return f"{host} · {self.model}"

    def _draw_speed_slider(self):
        cy = self.height - 220  # Vertical center of slider row

        label = f"SPEED {int(self.speech_rate * 100)}%"
        label_surf = self.font_small.render(label, True, C["dim"])
        self.screen.blit(label_surf, (18, cy - label_surf.get_height() // 2))

        track_x = 18 + label_surf.get_width() + 10
        track_x_end = self.width - 18
        track_w = track_x_end - track_x

        # Store hit rect for mouse events (slightly taller than visual track)
        self._slider_track = pygame.Rect(track_x, cy - 8, track_w, 16)

        # Track background
        pygame.draw.rect(self.screen, C["border"], (track_x, cy - 2, track_w, 4), border_radius=2)

        # Filled portion (left = 50%, right = 200%)
        fraction = (self.speech_rate - 0.5) / 1.5
        filled_w = int(fraction * track_w)
        if filled_w > 0:
            pygame.draw.rect(self.screen, C["ring_idle"], (track_x, cy - 2, filled_w, 4), border_radius=2)

        # Handle
        handle_x = track_x + filled_w
        pygame.draw.circle(self.screen, C["panel"], (handle_x, cy), 8)
        pygame.draw.circle(self.screen, C["ring_idle"], (handle_x, cy), 8, 2)

        # End labels
        lo_surf = self.font_small.render("50%", True, C["sys"])
        hi_surf = self.font_small.render("200%", True, C["sys"])
        self.screen.blit(lo_surf, (track_x, cy + 10))
        self.screen.blit(hi_surf, (track_x_end - hi_surf.get_width(), cy + 10))

    def _draw_mic_button(self):
        cfg = {
            "loading":      (C["btn_idle"], C["ring_idle"],    "…",  "loading models"),
            "idle":         (C["btn_idle"], C["ring_idle"],    "🎙", "click to speak"),
            "brain_offline":(C["btn_idle"], (180, 50, 50),    "✗",  "brain offline…"),
            "recording":    (C["btn_rec"],  C["ring_rec"],    "⏹", "click to send"),
            "processing":   (C["btn_proc"], C["ring_proc"],   "⋯",  "processing…"),
            "speaking":     (C["btn_talk"], C["ring_talk"],   "♪",  "speaking…"),
        }
        effective_state = self.state
        if self.state == "idle" and not self.brain_connected:
            effective_state = "brain_offline"
        bg, ring, symbol, hint = cfg.get(effective_state, cfg["idle"])
        self.hint_msg = hint
        
        # Glow rings
        for i in range(10, 0, -2):
            pygame.draw.circle(self.screen, ring, self.btn_mic_rect.center, 
                             65 + i, 1)
        
        # Main circle
        pygame.draw.circle(self.screen, bg, self.btn_mic_rect.center, 40, 0)
        pygame.draw.circle(self.screen, ring, self.btn_mic_rect.center, 40, 2)
        
        # Icon
        symbol_surf = self.font_title.render(symbol, True, ring)
        self.screen.blit(symbol_surf, 
                        (self.btn_mic_rect.centerx - symbol_surf.get_width() // 2,
                         self.btn_mic_rect.centery - symbol_surf.get_height() // 2))

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_models(self):
        try:
            self._set_status("Loading Whisper (base multilingual)…")
            self.whisper = WhisperModel("base", device="cpu", compute_type="int8")

            for lang, (onnx, jsn) in VOICE_FILES.items():
                self._set_status(f"Loading {LANG_FLAG[lang]} voice…")
                self.voices[lang] = PiperVoice.load(
                    os.path.join(VOICES_DIR, onnx),
                    config_path=os.path.join(VOICES_DIR, jsn),
                    use_cuda=False,
                )

            self.llm = OpenAI(base_url=self.brain_url, api_key="lm-studio")
            self.conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

            # Open always-on input stream
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                callback=self._audio_cb,
            )
            self.stream.start()

            self._set_state("idle")
            self.lang_badge = self._brain_label()
            self._log("AI", "en",
                      "Bonjour ! I'm your French tutor. Speak in English or French "
                      "— I'll match your language automatically.")
            if self.brain_connected:
                self._set_status(self._ready_status())
            else:
                self._set_status(self._ready_status())
        except Exception as e:
            self._set_status(f"Startup error: {e}")

    # ── Brain watchdog ────────────────────────────────────────────────────────
    def _brain_watchdog(self):
        """Polls LM Studio every 3 s; updates brain_connected and notifies on state change."""
        prev = None
        while True:
            connected = self._brain_api_ready(timeout=3)

            if connected != prev:
                self.brain_connected = connected
                if self.state != "loading":
                    if connected:
                        self._log_sys("Brain reconnected!")
                        self._set_status(self._ready_status())
                    else:
                        self._log_sys("Brain went offline.")
                        self._set_status(self._ready_status())
                prev = connected

            time.sleep(3)

    def _brain_api_ready(self, timeout=3):
        try:
            with urllib.request.urlopen(self.brain_url + "/models", timeout=timeout) as r:
                return r.status == 200
        except Exception:
            return False

    def _wait_for_brain_api(self, seconds=45):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._brain_api_ready(timeout=3):
                self.brain_connected = True
                self.llm = OpenAI(base_url=self.brain_url, api_key="lm-studio")
                self._set_status(self._ready_status())
                return True
            time.sleep(1)
        return False

    def _start_brain(self):
        if self.brain_starting:
            return
        threading.Thread(target=self._start_brain_worker, daemon=True).start()

    def _start_brain_worker(self):
        self.brain_starting = True
        try:
            command = self.config.get("brain_start_command") or []
            start_url = (self.config.get("brain_start_url") or "").strip()

            if isinstance(command, str):
                command = shlex.split(command)

            if command:
                self._log_sys("Starting Brain with configured command...")
                self._set_status("Starting Brain...")
                self._log_debug(f"brain start command: {command!r}")
                proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
                if proc.returncode == 0:
                    self._log_debug(f"brain start stdout: {proc.stdout.strip()}")
                    self._log_debug(f"brain start stderr: {proc.stderr.strip()}")
                    self._log_sys("Brain start command finished. Checking LM Studio...")
                    self._set_status("Checking Brain API...")
                    if self._wait_for_brain_api():
                        self._log_sys("Brain is online. Ready to chat.")
                    else:
                        self._log_sys("Brain command finished, but the API did not come online.")
                        self._set_status("Brain start finished, but API is still offline")
                else:
                    err = (proc.stderr or proc.stdout or "").strip()
                    self._log_sys(f"Brain start command failed: {err or proc.returncode}")
                    self._set_status("Brain start command failed")
                return

            if start_url:
                method = (self.config.get("brain_start_url_method") or "POST").upper()
                self._log_sys("Sending Brain start request...")
                self._set_status("Sending Brain start request...")
                req = urllib.request.Request(start_url, method=method)
                with urllib.request.urlopen(req, timeout=15) as r:
                    r.read()
                self._log_sys("Brain start request sent. Checking LM Studio...")
                self._set_status("Checking Brain API...")
                if self._wait_for_brain_api():
                    self._log_sys("Brain is online. Ready to chat.")
                else:
                    self._log_sys("Brain start request sent, but the API did not come online.")
                    self._set_status("Brain start request sent, but API is still offline")
                return

            self._log_sys("No Brain start command is configured yet.")
            self._set_status("Edit voicechat_config.json to configure START BRAIN")
        except subprocess.TimeoutExpired:
            self._log_sys("Brain start command timed out after 60 seconds.")
            self._set_status("Brain start timed out")
        except Exception as e:
            self._log_sys(f"Brain start failed: {e}")
            self._set_status(f"Brain start failed: {e}")
        finally:
            self.brain_starting = False

    def _log_sys(self, text):
        self.transcript.append(("SYS", "en", text, False))
        self._scroll_to_bottom = True

    def _log_debug(self, msg):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {msg}\n")
        except Exception:
            pass

    # ── Audio ─────────────────────────────────────────────────────────────────
    def _audio_cb(self, indata, frames, time, status):
        if self.state == "recording":
            self.audio_chunks.append(indata.copy())

    def _start_rec(self):
        self.audio_chunks = []
        self._set_state("recording")
        self._set_status("Listening… click again when done")

    def _stop_rec(self):
        chunks = self.audio_chunks[:]
        self._set_state("processing")
        self._set_status("Transcribing…")
        threading.Thread(target=self._process, args=(chunks,), daemon=True).start()

    # ── Pipeline ──────────────────────────────────────────────────────────────
    def _process(self, chunks):
        wav_path = None
        try:
            if not chunks:
                self._set_status("No audio recorded — try again")
                time.sleep(1.5)
                self._set_state("idle")
                self._set_status(self._ready_status())
                return

            audio = np.concatenate(chunks, axis=0)
            if audio.shape[0] < SAMPLE_RATE * 0.4:
                self._set_status("Too short — hold longer next time")
                time.sleep(1.5)
                self._set_state("idle")
                self._set_status(self._ready_status())
                return

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            sf.write(wav_path, audio, SAMPLE_RATE)

            segments, info = self.whisper.transcribe(
                wav_path, beam_size=5, language=None
            )
            user_text = " ".join(s.text.strip() for s in segments).strip()
            detected = info.language if info.language in ("en", "fr") else "en"
            self.last_lang = detected

            if not user_text:
                self._set_status("No speech detected — try again")
                time.sleep(1.5)
                self._set_state("idle")
                self._set_status(self._ready_status())
                return

            self._log("You", detected, user_text)
            flag = LANG_FLAG[detected]
            self.lang_badge = f"Detected: {flag}"

            # Quit phrase check
            quit_words = {"quit", "exit", "quitter", "arrêter", "stop"}
            if any(w in user_text.lower() for w in quit_words):
                self._speak("Au revoir !", "fr")
                self.running = False
                return

            # Build a per-turn directive so the model knows exactly what to do
            lang_directive = _build_lang_directive(user_text, detected)
            self.conversation.append({
                "role": "user",
                "content": f"{user_text}  [{lang_directive}]",
            })

            if not self.brain_connected:
                self._log_sys("Brain is offline. Press START BRAIN or wait for reconnect.")
                self._set_status("Brain offline — cannot send message yet")
                return

            self._set_status("Thinking on thebrain…")
            resp = self.llm.chat.completions.create(
                model=self.model,
                messages=self.conversation,
                max_tokens=300,
            )
            reply = resp.choices[0].message.content.strip()

            if not reply and hasattr(resp.choices[0].message, "reasoning_content"):
                reply = resp.choices[0].message.reasoning_content.strip()

            if not reply:
                reply = "Je ne sais pas." if detected == "fr" else "I'm not sure."

            self.conversation.append({"role": "assistant", "content": reply})
            self._trim_conversation()
            clean_reply = _strip_lang_tags(reply)
            ai_line_idx = self._log("AI", detected, clean_reply)

            self._set_state("speaking")
            self._set_status(f"Speaking in {flag}…")
            self._speak(reply, detected, ai_line_idx)

        except Exception as e:
            self._set_status(f"Error: {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)
            self._set_state("idle")
            self._set_status(self._ready_status())

    def _trim_conversation(self):
        if not self.conversation:
            return
        system = self.conversation[:1]
        recent = self.conversation[1:][-self.max_history_turns * 2:]
        self.conversation = system + recent

    def _speak(self, text, lang, line_idx=None):
        """Sentence-level synthesis (natural audio) + proportional word highlighting."""
        segments = _parse_lang_tags(text, default_lang=lang)
        self._log_debug(f"TTS start lang={lang} rate={self.speech_rate:.2f} text={_strip_lang_tags(text)!r}")

        # Build (voice, seg_text, [words]) per segment, skipping bare punctuation words
        parsed = []
        for seg_lang, seg_text in segments:
            voice = self.voices.get(seg_lang) or self.voices.get("en")
            words = [w for w in seg_text.split() if re.search(r"[A-Za-zÀ-öø-ÿ0-9]", w)]
            if words:
                parsed.append((voice, seg_text, words))

        if not parsed:
            return

        self.speaking_line_idx = line_idx
        wav_path = None

        try:
            SR = 22050
            syn_config = SynthesisConfig(length_scale=1.0 / self.speech_rate)
            seg_gap = np.zeros(int(SR * 0.08), dtype="float32")

            # ── Pass 1: synthesize each segment as a whole (natural prosody) ──
            seg_audios = []
            for voice, seg_text, _ in parsed:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    voice.synthesize_wav(seg_text, wf, syn_config=syn_config)
                buf.seek(0)
                audio, _ = sf.read(buf, dtype="float32")
                self._log_debug(f"TTS segment generated samples={len(audio)} text={seg_text!r}")
                seg_audios.append(audio)

            # ── Pass 2: synthesize each word alone → duration proportions only ──
            seg_word_samples = []
            for voice, _, words in parsed:
                word_samples = []
                for word in words:
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        voice.synthesize_wav(word, wf, syn_config=syn_config)
                    buf.seek(0)
                    audio, _ = sf.read(buf, dtype="float32")
                    word_samples.append(max(len(audio), 1))
                seg_word_samples.append(word_samples)

            # ── Build combined audio: segments interleaved with gaps ──
            parts = []
            for i, audio in enumerate(seg_audios):
                if audio.size > 0:
                    parts.append(audio)
                if i < len(seg_audios) - 1:
                    parts.append(seg_gap)
            if not parts:
                return
            combined = np.concatenate(parts)

            # ── Compute per-word highlight durations (proportional to word audio) ──
            gap_dur = len(seg_gap) / SR
            word_durations = []
            for seg_i, (_, _, words) in enumerate(parsed):
                seg_dur = len(seg_audios[seg_i]) / SR
                samples = seg_word_samples[seg_i]
                total = sum(samples)
                for w_i, s in enumerate(samples):
                    dur = seg_dur * s / total
                    # absorb the inter-segment gap into the last word of this segment
                    if w_i == len(samples) - 1 and seg_i < len(parsed) - 1:
                        dur += gap_dur
                    word_durations.append(dur)

            # ── Play combined audio ──
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = f.name
            sf.write(wav_path, combined, SR)
            file_size = os.path.getsize(wav_path)
            duration = len(combined) / SR
            self._log_debug(f"TTS wav ready path={wav_path} bytes={file_size} duration={duration:.2f}s")

            proc = self._play_tts_wav(wav_path)

            for word_idx, dur in enumerate(word_durations):
                self.speaking_word_idx = word_idx
                time.sleep(dur)

            if proc:
                proc.wait(timeout=30)
                if proc.returncode != 0:
                    stderr = proc.stderr.read() if proc.stderr else b""
                    err = stderr.decode(errors="replace").strip()
                    self._log_debug(f"TTS playback failed returncode={proc.returncode} err={err}")

        except Exception as e:
            self._log_debug(f"TTS exception: {e}")
            print(f"[TTS] Exception: {e}", file=sys.stderr)
            self._set_status(f"TTS error: {e}")
        finally:
            self.speaking_line_idx = None
            self.speaking_word_idx = None
            if wav_path and os.path.exists(wav_path):
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    def _play_tts_wav(self, wav_path):
        for player in ("paplay", "aplay"):
            try:
                proc = subprocess.Popen([player, wav_path],
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE)
                time.sleep(0.15)
                if proc.poll() is None or proc.returncode == 0:
                    self._log_debug(f"TTS playback started with {player}")
                    return proc
                stderr = proc.stderr.read() if proc.stderr else b""
                err = stderr.decode(errors="replace").strip()
                self._log_debug(f"TTS {player} failed immediately: {err}")
            except FileNotFoundError:
                self._log_debug(f"TTS player not found: {player}")
            except Exception as e:
                self._log_debug(f"TTS {player} exception: {e}")

        try:
            audio, sr = sf.read(wav_path, dtype="float32")
            self._log_debug(f"TTS falling back to sounddevice samples={len(audio)} sr={sr}")
            sd.play(audio, sr, blocking=False)
        except Exception as e:
            self._log_debug(f"TTS sounddevice fallback failed: {e}")
            self._set_status(f"TTS playback failed: {e}")
        return None

    def _play_beep(self):
        sr = 48000
        freq = 440
        t = np.linspace(0, 1, sr, endpoint=False)
        tone = (np.sin(2 * np.pi * freq * t) * 0.3).astype("float32")
        
        while self.beeping:
            try:
                sd.play(tone, sr, blocking=False)  # Uses default device
            except Exception as e:
                self._set_status(f"Beep error: {e}")
                break
            time.sleep(1.1)  # 1 second tone + 0.1s gap

    # ── Transcript ────────────────────────────────────────────────────────────
    def _log(self, speaker, lang, text):
        self.transcript.append((speaker, lang, text, speaker == "AI"))
        self._scroll_to_bottom = True
        return len(self.transcript) - 1

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _set_state(self, state):
        self.state = state

    def _set_status(self, msg):
        self.status_msg = msg

    def _ready_status(self):
        if self.brain_connected:
            return "Ready — click the mic to speak"
        return "Brain offline — press START BRAIN or wait for reconnect"

    def _update_speech_rate(self, mouse_x):
        if self._slider_track is None:
            return
        fraction = (mouse_x - self._slider_track.x) / self._slider_track.width
        fraction = max(0.0, min(1.0, fraction))
        self.speech_rate = 0.5 + fraction * 1.5  # maps 0→50%, 1→200%

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self._slider_track and self._slider_track.collidepoint(event.pos):
                self.slider_dragging = True
                self._update_speech_rate(event.pos[0])
            elif self.start_btn_rect.collidepoint(event.pos):
                if self.brain_connected:
                    self._set_status("Brain is already online")
                else:
                    self._start_brain()
            elif self.btn_mic_rect.collidepoint(event.pos):
                if self.state == "idle" and not self.brain_connected:
                    self._set_status("Brain is offline — waiting to reconnect…")
                elif self.state == "idle":
                    self._start_rec()
                elif self.state == "recording":
                    self._stop_rec()
            elif self.test_btn_rect.collidepoint(event.pos):
                self.beeping = True
                threading.Thread(target=self._play_beep, daemon=True).start()
        elif event.type == pygame.MOUSEMOTION:
            if self.slider_dragging:
                self._update_speech_rate(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.slider_dragging = False
            if self.beeping:
                self.beeping = False
                sd.stop()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if self.state == "idle" and not self.brain_connected:
                    self._set_status("Brain is offline — waiting to reconnect…")
                elif self.state == "idle":
                    self._start_rec()
                elif self.state == "recording":
                    self._stop_rec()
            elif event.key == pygame.K_UP:
                self.scroll_offset = max(0, self.scroll_offset - 40)
            elif event.key == pygame.K_DOWN:
                self.scroll_offset = min(self.max_scroll, self.scroll_offset + 40)
            elif event.key == pygame.K_PAGEUP:
                self.scroll_offset = max(0, self.scroll_offset - 200)
            elif event.key == pygame.K_PAGEDOWN:
                self.scroll_offset = min(self.max_scroll, self.scroll_offset + 200)
            elif event.key == pygame.K_END:
                self.scroll_offset = self.max_scroll
            elif event.key == pygame.K_HOME:
                self.scroll_offset = 0
        elif event.type == pygame.MOUSEWHEEL:
            self.scroll_offset = max(0, min(self.max_scroll,
                                            self.scroll_offset - event.y * 30))
        elif event.type == pygame.VIDEORESIZE:
            self.width, self.height = event.size
            self.btn_mic_rect = pygame.Rect(self.width // 2 - 65,
                                           self.height - 200, 130, 130)
            self.start_btn_rect = pygame.Rect(self.width // 2 - 170,
                                             self.height - 60, 160, 40)
            self.test_btn_rect = pygame.Rect(self.width // 2 + 10,
                                            self.height - 60, 160, 40)

    def run(self):
        self.running = True
        while self.running:
            for event in pygame.event.get():
                self.handle_event(event)
            
            self.render()
            self.clock.tick(60)

        pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    app = VoiceChatApp()
    app.run()


if __name__ == "__main__":
    main()
