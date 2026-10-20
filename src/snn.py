"""Rede Neural de Pulsos (SNN) em PyTorch puro.

Neuronios LIF (Leaky Integrate-and-Fire) com surrogate gradient
para treinamento via backpropagation through time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeFn(torch.autograd.Function):
    """Heaviside com surrogate gradient (arctan)."""

    @staticmethod
    def forward(ctx, v_mem, v_th=1.0):
        ctx.save_for_backward(v_mem)
        ctx.v_th = v_th
        return (v_mem >= v_th).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v_mem,) = ctx.saved_tensors
        # surrogate: derivada de (1/pi) * arctan(pi * x)
        x = torch.pi * (v_mem - ctx.v_th)
        grad = grad_out / (1.0 + x * x)
        return grad, None


def spike(v_mem, v_th=1.0):
    return SpikeFn.apply(v_mem, v_th)


class LIF(nn.Module):
    """Neuronio LIF: integra entrada, dispara spike, reseta."""

    def __init__(self, beta=0.85, v_th=1.0):
        super().__init__()
        self.beta = beta
        self.v_th = v_th

    def forward(self, x, mem):
        mem = self.beta * mem + x
        s = spike(mem, self.v_th)
        mem = mem * (1.0 - s)  # reset por subtracao
        return s, mem

    def init_mem(self, like):
        return torch.zeros_like(like)


class SpikingCNN(nn.Module):
    """Classificador de caracteres 28x28 -> num_classes via spikes.

    Input encoding: a imagem e injetada como corrente constante
    durante `num_steps` passos de tempo (direct encoding).
    Readout: spike count da ultima camada.
    """

    def __init__(self, num_classes=36, num_steps=12, beta=0.85):
        super().__init__()
        self.num_steps = num_steps

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.lif1 = LIF(beta)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.lif2 = LIF(beta)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.lif3 = LIF(beta)
        self.fc2 = nn.Linear(256, num_classes)
        self.lif4 = LIF(beta, v_th=1e9)  # readout: so acumula membrana

    def forward(self, x, record=False):
        B = x.size(0)
        m1 = torch.zeros(B, 16, 28, 28, device=x.device)
        m2 = torch.zeros(B, 32, 14, 14, device=x.device)
        m3 = torch.zeros(B, 256, device=x.device)
        m4 = torch.zeros(B, self.fc2.out_features, device=x.device)

        rec = {"s1": [], "s2": [], "s3": [], "m4": []} if record else None
        mem4_sum = 0.0
        for _ in range(self.num_steps):
            s1, m1 = self.lif1(self.conv1(x), m1)
            s, m2 = self.lif2(self.conv2(F.max_pool2d(s1, 2)), m2)
            s3, m3 = self.lif3(self.fc1(F.max_pool2d(s, 2).flatten(1)), m3)
            _, m4 = self.lif4(self.fc2(s3), m4)
            mem4_sum = mem4_sum + m4
            if record:
                rec["s1"].append(s1)
                rec["s2"].append(s)
                rec["s3"].append(s3)
                rec["m4"].append(m4)
        out = mem4_sum / self.num_steps
        return (out, rec) if record else out


CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
