#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Models for the LEO SEU suite.

1. toy50          50 float32 weights as Linear(50, 1, bias=False) so the
                  tensor is dim>=2 (flippable). Sign-agreement accuracy.
2. mnist_linear   Flatten 784 → 128 → 10.
3. mnist_cnn      Conv32-Conv64-FC on MNIST (28×28, no upscale).
4. resnet18_t10   torchvision resnet18, 10-class, CIFAR-10 32×32.
                  Prefer ImageNet pretrained + freeze backbone, train last
                  layer. If the weight download fails: reduced resnet18
                  (conv1 3×3 stride 1, no maxpool) trained from scratch on
                  a CIFAR-10 subset. Still named resnet18_t10.
5. cifar_deep     3-stage CIFAR-10 CNN, ~300k params.

N_bits is counted from weight tensors (dim>=2) only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models

from radiation import collect_weight_params, n_bits_of, n_weight_elements

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
if not os.path.isdir(DATA_DIR):
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

MODEL_NAMES = ("toy50", "mnist_linear", "mnist_cnn", "resnet18_t10", "cifar_deep")


@dataclass
class ModelSpec:
    name: str
    n_classes: int
    input_kind: str  # toy | mnist | cifar
    chance_acc: float  # percent
    build: Callable[[], nn.Module]


# ===========================================================================
# Architectures
# ===========================================================================
class Toy50(nn.Module):
    """Course toy: 50 float32 weights. Stored as (1, 50) so dim>=2 is flippable."""

    def __init__(self, std: float = 0.5) -> None:
        super().__init__()
        self.fc = nn.Linear(50, 1, bias=False)
        nn.init.normal_(self.fc.weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 50) → (B, 1)
        return self.fc(x)


class MNISTLinear(nn.Module):
    """784-128-10 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class MNISTCNN(nn.Module):
    """Small Conv-Conv-FC on 28×28 MNIST. ~220k weights."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))  # 14×14
        x = self.pool(F.relu(self.conv2(x)))  # 7×7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CifarDeep(nn.Module):
    """3 conv stages on 32×32 CIFAR-10, ~307k params. Deeper-but-testable."""

    def __init__(self) -> None:
        super().__init__()
        self.s1 = nn.Sequential(  # 32×32
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.s2 = nn.Sequential(  # 16×16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.s3 = nn.Sequential(  # 8×8
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 4×4
        )
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s3(self.s2(self.s1(x)))
        return self.fc(x.view(x.size(0), -1))


def _reduced_resnet18() -> nn.Module:
    """CIFAR-friendly resnet18: 3×3 conv1 stride 1, no maxpool, 10 classes."""
    m = models.resnet18(weights=None, num_classes=10)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def build_resnet18_t10(try_pretrained: bool = True) -> tuple[nn.Module, str]:
    """Return (model, variant_note). variant is 'pretrained_fc' or 'scratch_reduced'."""
    if try_pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            m = models.resnet18(weights=weights)
            in_f = int(m.fc.in_features)
            m.fc = nn.Linear(in_f, 10)
            for name, p in m.named_parameters():
                if not name.startswith("fc"):
                    p.requires_grad = False
            # Keep 32×32 input — do NOT upscale to 224 (CPU constraint).
            return m, "pretrained_fc_32x32_no_upscale"
        except Exception as exc:  # noqa: BLE001 — download / SSL / offline
            print(f"[resnet18_t10] pretrained download failed ({exc!r}); "
                  "falling back to reduced resnet18 from scratch")
    return _reduced_resnet18(), "scratch_reduced_conv1_3x3_no_maxpool"


# Track which resnet variant the process actually built (for NOTES.md).
RESNET_VARIANT = "unbuilt"


def build_model(name: str, try_pretrained: bool = True) -> nn.Module:
    global RESNET_VARIANT
    name = name.lower().strip()
    if name == "toy50":
        return Toy50()
    if name == "mnist_linear":
        return MNISTLinear()
    if name == "mnist_cnn":
        return MNISTCNN()
    if name == "cifar_deep":
        return CifarDeep()
    if name == "resnet18_t10":
        m, variant = build_resnet18_t10(try_pretrained=try_pretrained)
        RESNET_VARIANT = variant
        return m
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")


def n_classes_of(name: str) -> int:
    return 2 if name == "toy50" else 10


def chance_acc_of(name: str) -> float:
    """Percent. 1/n_classes * 100, used when a run crashes with no finite acc."""
    return 100.0 / n_classes_of(name)


def describe_model(model: nn.Module, flip_bias: bool = False) -> dict:
    params = collect_weight_params(model, flip_bias=flip_bias)
    n_w = n_weight_elements(params)
    n_b = n_bits_of(params)
    n_bias = sum(int(p.numel()) for n, p in model.named_parameters() if p.dim() == 1)
    n_all = sum(int(p.numel()) for p in model.parameters())
    return {
        "n_weight_elements": n_w,
        "n_bits": n_b,
        "n_bias_elements": n_bias,
        "n_all_params": n_all,
        "n_tensors": len(params),
    }


# ===========================================================================
# Data
# ===========================================================================
@dataclass
class TensorDataset:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    extra: dict | None = None


def _cifar_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


def load_mnist(n_train: int, n_test: int) -> TensorDataset:
    os.makedirs(DATA_DIR, exist_ok=True)
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True)
    x_train = train_ds.data[:n_train].unsqueeze(1).float() / 255.0  # (N,1,28,28)
    y_train = train_ds.targets[:n_train].long()
    x_test = test_ds.data[:n_test].unsqueeze(1).float() / 255.0
    y_test = test_ds.targets[:n_test].long()
    return TensorDataset(x_train, y_train, x_test, y_test)


def load_cifar10(n_train: int, n_test: int) -> TensorDataset:
    os.makedirs(DATA_DIR, exist_ok=True)
    train_ds = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)
    x_tr = torch.from_numpy(np.asarray(train_ds.data[:n_train])).permute(0, 3, 1, 2).float() / 255.0
    y_tr = torch.tensor(train_ds.targets[:n_train], dtype=torch.long)
    x_te = torch.from_numpy(np.asarray(test_ds.data[:n_test])).permute(0, 3, 1, 2).float() / 255.0
    y_te = torch.tensor(test_ds.targets[:n_test], dtype=torch.long)
    return TensorDataset(_cifar_normalize(x_tr), y_tr, _cifar_normalize(x_te), y_te)


def make_toy_data(
    n_train: int = 2000,
    n_test: int = 500,
    seed: int = 2026,
) -> TensorDataset:
    """Synthetic linear task. Teacher weights ~ N(0, 0.5^2); y = sign(W·x)."""
    rng = np.random.default_rng(int(seed))
    teacher = rng.normal(0.0, 0.5, size=(1, 50)).astype(np.float32)
    x_tr = rng.standard_normal((n_train, 50), dtype=np.float32)
    x_te = rng.standard_normal((n_test, 50), dtype=np.float32)
    y_tr = (np.sign(x_tr @ teacher.T).ravel() >= 0).astype(np.int64)
    y_te = (np.sign(x_te @ teacher.T).ravel() >= 0).astype(np.int64)
    return TensorDataset(
        torch.from_numpy(x_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(x_te),
        torch.from_numpy(y_te),
        extra={"teacher": teacher},
    )


def load_data_for(model_name: str, n_train: int, n_test: int, toy_seed: int = 2026) -> TensorDataset:
    name = model_name.lower().strip()
    if name == "toy50":
        return make_toy_data(n_train=max(n_train, 500), n_test=max(n_test, 200), seed=toy_seed)
    if name in ("mnist_linear", "mnist_cnn"):
        return load_mnist(n_train, n_test)
    if name in ("resnet18_t10", "cifar_deep"):
        return load_cifar10(n_train, n_test)
    raise ValueError(model_name)


def default_budget(model_name: str) -> dict:
    """Starting train/test sizes and epochs; paper_sweep may shrink after a pilot."""
    name = model_name.lower().strip()
    if name == "toy50":
        return dict(n_train=2000, n_test=500, epochs=8, ckpt_epochs=8, batch_size=64, lr=0.05, momentum=0.0)
    if name == "mnist_linear":
        return dict(n_train=60_000, n_test=10_000, epochs=2, ckpt_epochs=2, batch_size=128, lr=0.05, momentum=0.9)
    if name == "mnist_cnn":
        return dict(n_train=60_000, n_test=10_000, epochs=2, ckpt_epochs=2, batch_size=128, lr=0.05, momentum=0.9)
    if name == "resnet18_t10":
        # train-mode radiation: 1 epoch (last layer). Infer clean ckpt: a few more epochs
        # of the cheap frozen-backbone head so baseline acc is not chance-level.
        return dict(n_train=10_000, n_test=2_000, epochs=1, ckpt_epochs=8, batch_size=64, lr=0.01, momentum=0.9)
    if name == "cifar_deep":
        return dict(n_train=10_000, n_test=2_000, epochs=1, ckpt_epochs=6, batch_size=64, lr=0.05, momentum=0.9)
    raise ValueError(name)
