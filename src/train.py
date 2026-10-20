"""Treinamento da SNN. Uso: python -m src.train --epochs 5 --emnist"""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from .snn import SpikingCNN, CHARSET
from .dataset import SyntheticCharDataset, load_emnist


class EMNISTMapped(Dataset):
    """EMNIST byclass filtrado p/ labels 0-35 (0-9 + A-Z), alinhado ao CHARSET."""

    def __init__(self, base):
        self.idx = [i for i in range(len(base)) if base.targets[i].item() < len(CHARSET)]
        self.base = base

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--samples-per-class", type=int, default=2000)
    ap.add_argument("--emnist", action="store_true", help="usa EMNIST publico em vez de sintetico")
    ap.add_argument("--cpu", action="store_true", help="forca CPU")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"device={device}")

    if args.emnist:
        train_base, test_base = load_emnist("byclass")
        train_set, test_set = EMNISTMapped(train_base), EMNISTMapped(test_base)
    else:
        ds = SyntheticCharDataset(args.samples_per_class)
        test_size = len(CHARSET) * 100
        train_set, test_set = random_split(ds, [len(ds) - test_size, test_size],
                                           generator=torch.Generator().manual_seed(0))
    print(f"train={len(train_set)} test={len(test_set)}")

    train_dl = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    test_dl = DataLoader(test_set, batch_size=512, num_workers=2)

    model = SpikingCNN(num_classes=len(CHARSET), num_steps=args.steps).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        model.train()
        t0, total, correct = time.time(), 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
        sched.step()

        model.eval()
        tcorrect, ttotal = 0, 0
        with torch.no_grad():
            for x, y in test_dl:
                x, y = x.to(device), y.to(device)
                tcorrect += (model(x).argmax(1) == y).sum().item()
                ttotal += y.size(0)
        print(f"epoch {ep}: train_acc={correct/total:.4f} "
              f"test_acc={tcorrect/ttotal:.4f} ({time.time()-t0:.1f}s)")

    out = "checkpoints/snn_ocr.pt"
    torch.save({"model": model.state_dict(), "charset": CHARSET,
                "num_steps": args.steps}, out)
    print(f"salvo em {out}")


if __name__ == "__main__":
    main()
