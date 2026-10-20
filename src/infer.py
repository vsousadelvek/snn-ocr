"""Pipeline de inferencia: imagem -> segmentar caracteres -> SNN -> texto."""
import numpy as np
import torch
from PIL import Image, ImageOps

from .snn import SpikingCNN, CHARSET


def binarize(gray):
    """Binariza com Otsu, garantindo texto branco sobre fundo preto."""
    arr = np.asarray(gray).astype(np.float32)
    hist = np.bincount((arr).astype(np.uint8).ravel(), minlength=256)
    total = arr.size
    sum_all = np.dot(np.arange(256), hist)
    sum_b, w_b, best_th, best_var = 0.0, 0, 0, 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0 or w_b == total:
            continue
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_all - sum_b) / (total - w_b)
        var = w_b * (total - w_b) * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_th = var, t
    binary = (arr > best_th).astype(np.uint8)
    if binary.mean() > 0.5:  # mais fundo branco que texto -> inverter
        binary = 1 - binary
    return binary


def segment(binary):
    """Segmenta caracteres por projecao vertical com fallback p/ componentes."""
    h, w = binary.shape
    col = binary.sum(0)
    ink = col > 0.01 * h
    boxes, start = [], None
    min_gap = max(2, w // 200)
    gap = 0
    for x in range(w):
        if ink[x]:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                boxes.append((start, x - gap))
                start = None
    if start is not None:
        boxes.append((start, w))

    crops = []
    for x0, x1 in boxes:
        strip = binary[:, x0:x1]
        rows = np.where(strip.sum(1) > 0)[0]
        if len(rows) == 0 or (x1 - x0) < 3:
            continue
        y0, y1 = rows[0], rows[-1] + 1
        crops.append((x0, strip[y0:y1, :]))
    return crops


def to_28(crop):
    """Redimensiona mantendo aspecto e centraliza em 28x28."""
    img = Image.fromarray((crop * 255).astype(np.uint8))
    w, h = img.size
    scale = 20.0 / max(w, h)
    img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(img, ((28 - img.size[0]) // 2, (28 - img.size[1]) // 2))
    return torch.from_numpy(np.asarray(canvas)).float().unsqueeze(0) / 255.0


class OCREngine:
    def __init__(self, checkpoint="checkpoints/snn_ocr.pt", device=None):
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.charset = ckpt.get("charset", CHARSET)
        self.model = SpikingCNN(num_classes=len(self.charset),
                                num_steps=ckpt.get("num_steps", 12))
        self.model.load_state_dict(ckpt["model"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def predict_image(self, img):
        """img: PIL.Image qualquer -> (texto, confianca media)."""
        gray = ImageOps.grayscale(img)
        binary = binarize(gray)
        crops = sorted(segment(binary))
        if not crops:
            return "", 0.0
        batch = torch.stack([to_28(c) for _, c in crops])
        with torch.no_grad():
            logits = self.model(batch.to(self.device))
            probs = torch.softmax(logits, dim=1)
            conf, idx = probs.max(1)

        # insere espaco quando o gap horizontal e grande demais
        widths = [c.shape[1] for _, c in crops]
        med_w = float(np.median(widths))
        parts, prev_end = [], None
        for (x0, crop), i in zip(crops, idx.tolist()):
            if prev_end is not None and (x0 - prev_end) > 1.2 * med_w:
                parts.append(" ")
            parts.append(self.charset[i])
            prev_end = x0 + crop.shape[1]
        return "".join(parts), float(conf.mean())


if __name__ == "__main__":
    import sys
    engine = OCREngine()
    text, conf = engine.predict_image(Image.open(sys.argv[1]))
    print(f"'{text}' (conf={conf:.2f})")
