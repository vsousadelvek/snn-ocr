"""API FastAPI: POST /ocr com uma imagem -> {text, confidence, chars}."""
import io
import time

from fastapi import FastAPI, File, UploadFile
from PIL import Image

from .infer import OCREngine

app = FastAPI(title="SNN-OCR", description="OCR com Spiking Neural Network proprietaria")
engine: OCREngine | None = None


@app.on_event("startup")
def load_model():
    global engine
    engine = OCREngine()


@app.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    img = Image.open(io.BytesIO(await file.read()))
    text, conf = engine.predict_image(img)
    return {
        "text": text,
        "confidence": round(conf, 4),
        "chars": len(text),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model": "SpikingCNN", "device": engine.device}
