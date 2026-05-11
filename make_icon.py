#!/usr/bin/env python3
"""Generate the VoiceChat app icon as a 256x256 PNG."""
from PIL import Image, ImageDraw, ImageFilter
import math, os

SIZE = 256
OUT = os.path.join(os.path.dirname(__file__), "voicechat_icon.png")

img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

# ── Rounded square background gradient ──────────────────────────────────────
bg = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
bg_draw = ImageDraw.Draw(bg)
for i in range(SIZE // 2):
    t = i / (SIZE // 2)
    r = int(18  + t * 40)
    g = int(8   + t * 10)
    b = int(55  + t * 80)
    bg_draw.rounded_rectangle([i, i, SIZE - i, SIZE - i], radius=48 - i // 3,
                               fill=(r, g, b, 255))
img.alpha_composite(bg)

# ── Outer glow ring ───────────────────────────────────────────────────────────
glow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
for i in range(18):
    alpha = int(90 * (1 - i / 18))
    gd.ellipse([58 + i, 52 + i, 198 - i, 192 - i], outline=(160, 80, 255, alpha), width=2)
glow = glow.filter(ImageFilter.GaussianBlur(4))
img.alpha_composite(glow)

# ── Microphone body ───────────────────────────────────────────────────────────
draw = ImageDraw.Draw(img)
mic_x, mic_top, mic_bot, mic_w = 128, 58, 148, 34
draw.rounded_rectangle(
    [mic_x - mic_w // 2, mic_top, mic_x + mic_w // 2, mic_bot],
    radius=17,
    fill=(220, 180, 255, 255),
)
# inner highlight
draw.rounded_rectangle(
    [mic_x - mic_w // 2 + 5, mic_top + 5, mic_x + mic_w // 2 - 14, mic_top + 28],
    radius=8,
    fill=(255, 255, 255, 60),
)

# ── Mic stand arc ────────────────────────────────────────────────────────────
for i in range(3):
    draw.arc([mic_x - 36 + i, mic_bot - 20 + i, mic_x + 36 - i, mic_bot + 44 - i],
             start=0, end=180, fill=(200, 160, 255, 220), width=4)
draw.line([mic_x, mic_bot + 28, mic_x, mic_bot + 46], fill=(200, 160, 255, 220), width=4)
draw.line([mic_x - 18, mic_bot + 46, mic_x + 18, mic_bot + 46],
          fill=(200, 160, 255, 220), width=4)

# ── Sound-wave arcs (right side) ─────────────────────────────────────────────
cx, cy = 128, 103
for j, (r, a) in enumerate([(52, 180), (68, 160), (86, 130)]):
    alpha = 220 - j * 40
    for i in range(3):
        draw.arc([cx - r + i, cy - r + i, cx + r - i, cy + r - i],
                 start=-50 - j * 12, end=50 + j * 12,
                 fill=(120, 220, 255, alpha), width=3 - i)

# ── AI sparkle dots ───────────────────────────────────────────────────────────
sparkles = [
    (168, 72, 6, (255, 200, 80, 230)),
    (88,  68, 4, (255, 200, 80, 180)),
    (174, 148, 5, (180, 255, 200, 210)),
    (82,  148, 3, (180, 255, 200, 160)),
    (155, 190, 4, (160, 180, 255, 190)),
]
for sx, sy, sr, sc in sparkles:
    draw.ellipse([sx - sr, sy - sr, sx + sr, sy + sr], fill=sc)

# ── "AI" label at bottom ─────────────────────────────────────────────────────
try:
    from PIL import ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
except Exception:
    font = ImageFont.load_default()
label = "VOICE · AI"
bbox = draw.textbbox((0, 0), label, font=font)
tw = bbox[2] - bbox[0]
draw.text(((SIZE - tw) // 2, 206), label, font=font, fill=(200, 160, 255, 220))

# ── Soft vignette ─────────────────────────────────────────────────────────────
vig = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
vd = ImageDraw.Draw(vig)
for i in range(40):
    a = int(60 * (i / 40) ** 2)
    vd.rounded_rectangle([i, i, SIZE - i, SIZE - i], radius=48 - i // 2,
                          outline=(0, 0, 0, a), width=1)
img.alpha_composite(vig)

img.save(OUT, "PNG")
print(f"Icon saved: {OUT}")
