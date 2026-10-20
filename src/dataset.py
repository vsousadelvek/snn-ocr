"""Dados: gerador sintetico de caracteres + fallback EMNIST."""
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch.utils.data import Dataset

from .snn import CHARSET

FONT_DIRS = [r"C:\Windows\Fonts", "/usr/share/fonts", str(Path.home() / "fonts")]


def _list_fonts():
    for d in FONT_DIRS:
        p = Path(d)
        if p.exists():
            fonts = list(p.rglob("*.ttf"))
            if fonts:
                return fonts
    return []


def _valid_font(path):
    try:
        f = ImageFont.truetype(str(path), 20)
        f.getbbox("Ag")
        return True
    except Exception:
        return False


FONTS = [f for f in _list_fonts() if _valid_font(f)]


def render_char(char, size=28):
    """Renderiza um caractere com fonte/estilo/ruido aleatorios."""
    canvas = 48
    img = Image.new("L", (canvas, canvas), color=0)
    d = ImageDraw.Draw(img)

    font_size = random.randint(24, 34)
    try:
        font = ImageFont.truetype(str(random.choice(FONTS)), font_size) if FONTS else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    try:
        bbox = d.textbbox((0, 0), char, font=font)
    except Exception:
        font = ImageFont.load_default()
        bbox = d.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (canvas - w) // 2 - bbox[0] + random.randint(-4, 4)
    y = (canvas - h) // 2 - bbox[1] + random.randint(-4, 4)
    d.text((x, y), char, fill=random.randint(180, 255), font=font)

    img = img.rotate(random.uniform(-15, 15), resample=Image.BILINEAR, fillcolor=0)
    if random.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))

    arr = np.asarray(img).astype(np.float32) / 255.0
    arr += np.random.normal(0, random.uniform(0.0, 0.08), arr.shape)
    arr = np.clip(arr, 0, 1)

    img = Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    return img


class SyntheticCharDataset(Dataset):
    """Gera amostras em tempo real (infinito), uma imagem por caractere."""

    def __init__(self, samples_per_class=2000, size=28, chars=CHARSET):
        self.chars = chars
        self.samples_per_class = samples_per_class
        self.size = size
        self.n = len(chars) * samples_per_class

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        label = idx % len(self.chars)
        img = render_char(self.chars[label], self.size)
        x = torch.from_numpy(np.asarray(img)).float().unsqueeze(0) / 255.0
        return x, label


def load_emnist(split="balanced"):
    """Dataset publico EMNIST via torchvision (baixa automaticamente)."""
    from torchvision.datasets import EMNIST
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2)),  # EMNIST vem rotacionado
    ])
    return EMNIST(root="data", split=split, train=True, download=True, transform=tf), \
           EMNIST(root="data", split=split, train=False, download=True, transform=tf)
