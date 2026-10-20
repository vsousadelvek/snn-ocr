# SNN-OCR

OCR construído do zero com uma **Spiking Neural Network (SNN) proprietária** — neurônios LIF (Leaky Integrate-and-Fire) implementados em PyTorch puro, sem bibliotecas de SNN. Leve o suficiente para rodar rápido em CPU, treinável em minutos em GPU.

## Por que SNN?

Redes neurais de pulsos processam informação de forma **esparsa e orientada a eventos**, como o cérebro. Na prática: menos operações, menor consumo de energia e latência baixa — exatamente o que se quer em OCR embarcado/edge. O spike usa uma **surrogate gradient** (derivada de arctan) para permitir backpropagation through time.

## Arquitetura

```
Imagem 28x28 ──[direct encoding, T=12 passos]──> Conv(16) + LIF
            -> MaxPool -> Conv(32) + LIF -> MaxPool
            -> FC(256) + LIF -> FC(36) + LIF (readout por membrana)
            -> softmax -> 0-9 / A-Z
```

- **Input encoding:** a imagem entra como corrente constante durante 12 timesteps
- **Reset:** por subtração após cada spike
- **Readout:** média do potencial de membrana da camada final

## Pipeline de OCR completa

1. Binarização (Otsu implementado à mão, com auto-inversão)
2. Segmentação de caracteres por projeção vertical
3. Normalização para 28x28 (aspecto preservado)
4. Classificação pela SNN
5. Reconstrução do texto com detecção de espaços por gap horizontal

## Dados

- **Sintéticos:** gerador on-the-fly — ~260 fontes do sistema, rotação ±15°, blur, ruído gaussiano, offsets aleatórios (dataset "infinito", zero armazenamento)
- **Público:** flag `--emnist` treina no EMNIST byclass (labels 0–35 alinhados ao charset)

## Uso

```bash
pip install -r requirements.txt

# treinar (sintético)
python -m src.train --epochs 5

# treinar com EMNIST público
python -m src.train --emnist --epochs 10

# inferência em uma imagem
python -m src.infer demo/hello.png

# gerar imagem de teste
python demo/make_text_image.py "MEU TEXTO"

# API
python -m uvicorn src.api:app --port 8000
curl -X POST -F "file=@demo/hello.png" http://localhost:8000/ocr
# {"text":"HELLO2028","confidence":0.86,"chars":9,"latency_ms":~50}
```

## Resultados

| Métrica | Valor |
|---|---|
| Test accuracy (sintético) | ~94% |
| Parâmetros | ~440K |
| Treino | ~5 min (GPU), ~30 min (CPU) |
| Inferência | ms por caractere, batch de linha inteira |

## Estrutura

```
src/snn.py      -> LIF + SpikeFn (surrogate gradient) + SpikingCNN
src/dataset.py  -> gerador sintético + loader EMNIST
src/train.py    -> loop de treino
src/infer.py    -> binarização, segmentação, engine de OCR
src/api.py      -> FastAPI /ocr
```

## Próximos passos

- [ ] CTC loss para texto sem segmentação explícita
- [ ] Quantização dos pesos (event-driven inference real)
- [ ] Fine-tune em dados reais do domínio alvo
- [ ] Benchmark energético SNN vs CNN equivalente
