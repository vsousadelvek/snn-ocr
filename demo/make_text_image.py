"""Gera uma imagem de texto para testar a API."""
import sys
from PIL import Image, ImageDraw, ImageFont

text = sys.argv[1] if len(sys.argv) > 1 else "HELLO 2026"
out = sys.argv[2] if len(sys.argv) > 2 else "demo/text.png"
font = ImageFont.truetype("arial.ttf", 52)
w = int(font.getlength(text)) + 40
img = Image.new("L", (w, 90), 255)
ImageDraw.Draw(img).text((20, 15), text, fill=0, font=font)
img.save(out)
print(f"salvo: {out}")
