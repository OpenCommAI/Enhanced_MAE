# transformer_train_8TTI_to_X_output_length.py
import os
import math
import random
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm


# =========================
# 位置编码（正余弦）
# =========================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# =========================
# Transformer：按照 pred_tti 直接输出对应长度
# =========================
class TransformerRecon(nn.Module):
    """
    输入:
        x: (B, 2, 64, 32*history_tti)

    输出:
        pred: (B, 2, 64, 32*pred_tti)

    说明:
        不再使用 max_pred_tti。
        pred_tti 是多少，模型就直接输出多少个未来 TTI。
    """
    def __init__(
        self,
        in_ch=2,
        H=64,
        history_tti=8,
        pred_tti=1,
        d_model=128,
        nhead=4,
        num_layers=6,
        dim_feedforward=512,
        dropout=0.1,
        norm_first=True,
    ):
        super().__init__()

        assert history_tti >= 1, "history_tti must be >= 1"
        assert pred_tti >= 1, "pred_tti must be >= 1"

        self.in_ch = in_ch
        self.H = H
        self.history_tti = history_tti
        self.pred_tti = pred_tti

        self.W_in = 32 * history_tti
        self.W_out = 32 * pred_tti
        self.in_feat = in_ch * H

        self.input_proj = nn.Linear(self.in_feat, d_model)

        self.pos_encoding = SinusoidalPositionalEncoding(
            d_model=d_model,
            max_len=self.W_in
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 这里直接输出 32*pred_tti，不再固定到 max_pred_tti
        self.reduce_len = nn.AdaptiveAvgPool1d(self.W_out)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.in_feat),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.constant_(self.input_proj.bias, 0.0)

        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: (B, 2, 64, 32*history_tti)
        B, C, H, W = x.shape

        assert (C, H, W) == (self.in_ch, self.H, self.W_in), \
            f"expect (B,{self.in_ch},{self.H},{self.W_in}), got {tuple(x.shape)}"

        # (B, 2, 64, W_in) -> (B, W_in, 2, 64) -> (B, W_in, 128)
        seq = x.permute(0, 3, 1, 2).contiguous()
        seq = seq.view(B, self.W_in, C * H)

        seq = self.input_proj(seq)
        seq = self.pos_encoding(seq)

        seq_out = self.encoder(seq)

        # (B, W_in, d_model) -> (B, d_model, W_in)
        seq_out = seq_out.transpose(1, 2)

        # 直接压缩或扩展到 W_out = 32*pred_tti
        seq_out = self.reduce_len(seq_out)

        # (B, d_model, W_out) -> (B, W_out, d_model)
        seq_out = seq_out.transpose(1, 2).contiguous()

        # (B, W_out, d_model) -> (B, W_out, 128)
        step_feat = self.head(seq_out)

        # (B, W_out, 128) -> (B, W_out, 2, 64)
        step_feat = step_feat.view(
            B,
            self.W_out,
            self.in_ch,
            self.H
        )

        # (B, W_out, 2, 64) -> (B, 2, 64, W_out)
        pred = step_feat.permute(0, 2, 3, 1).contiguous()

        return pred


# =========================
# NMSE Loss
# =========================
class NMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-12, reduction: str = "mean"):
        super().__init__()

        assert reduction in ("none", "mean", "sum")

        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num = ((pred - target) ** 2).reshape(pred.size(0), -1).sum(dim=1)
        den = (target ** 2).reshape(target.size(0), -1).sum(dim=1) + self.eps

        nmse = num / den

        if self.reduction == "mean":
            return nmse.mean()
        if self.reduction == "sum":
            return nmse.sum()

        return nmse


# =========================
# 配置
# =========================
@dataclass
class Config:
    num1 = 8
    num2 = 25

    data_dir: str = fr"./data/{num2}dB_zscore_norm_8to1_dataset"

    # 输出路径
    out_root: str = "./outputs/transformer"

    history_tti: int = 8
    pred_tti: int = 1

    batch_size: int = 128
    num_workers: int = 1
    epochs: int = 300

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    amp: bool = True
    seed: int = 52
    save_every: int = 10

    warmup_epochs: int = 10
    eta_min: float = 1e-6

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    norm_first: bool = True

    device: str = "cuda"

cfg = Config()


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


def to_db(x: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(x), eps))


def resolve_device(device_cfg: str):
    if device_cfg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device_cfg}', but CUDA is unavailable.")

        if device_cfg == "cuda":
            return torch.device("cuda")

        try:
            idx = int(device_cfg.split(":")[1])
        except Exception as e:
            raise ValueError(f"Invalid CUDA device format: {device_cfg}") from e

        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device '{device_cfg}', "
                f"but only {torch.cuda.device_count()} CUDA device(s) are available."
            )

        return torch.device(device_cfg)

    if device_cfg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device config: {device_cfg}")


# =========================
# 数据
# =========================
def complex_to_2ch(x: np.ndarray):
    """
    输入:
        x: (N, 64, 32*k), complex

    输出:
        x: (N, 2, 64, 32*k), float32
    """
    assert x.ndim == 3, f"expect 3D array, got {x.ndim}D, shape={x.shape}"

    real = np.real(x)
    imag = np.imag(x)

    x = np.stack([real, imag], axis=1)

    return x.astype(np.float32)


def array_to_2ch(x: np.ndarray, name: str):
    """
    统一转成:
        (N, 2, 64, W)

    支持:
        1. complex 3D: (N, 64, W)
        2. real 3D   : (N, 64, W)，虚部补 0
        3. 4D        : (N, 2, 64, W)
        4. 4D        : (N, 64, W, 2)
    """
    if x.ndim == 4:
        if x.shape[1] == 2:
            return x.astype(np.float32)

        if x.shape[-1] == 2:
            x = np.transpose(x, (0, 3, 1, 2))
            return x.astype(np.float32)

        raise ValueError(f"{name} 是 4D，但无法识别通道维度，shape={x.shape}")

    if x.ndim == 3:
        if np.iscomplexobj(x):
            return complex_to_2ch(x)

        zeros = np.zeros_like(x)
        x = np.stack([x, zeros], axis=1)
        return x.astype(np.float32)

    raise ValueError(f"{name} 维度错误，期望 3D 或 4D，实际 shape={x.shape}")


def load_data(data_dir):
    train_inputs_path = os.path.join(data_dir, "train_inputs.npy")
    train_labels_path = os.path.join(data_dir, "train_labels.npy")
    test_inputs_path = os.path.join(data_dir, "test_inputs.npy")
    test_labels_path = os.path.join(data_dir, "test_labels.npy")

    for path in [
        train_inputs_path,
        train_labels_path,
        test_inputs_path,
        test_labels_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")

    Xtr = np.load(train_inputs_path)
    Ytr = np.load(train_labels_path)
    Xte = np.load(test_inputs_path)
    Yte = np.load(test_labels_path)

    print("Raw Xtr:", Xtr.shape, Xtr.dtype)
    print("Raw Ytr:", Ytr.shape, Ytr.dtype)
    print("Raw Xte:", Xte.shape, Xte.dtype)
    print("Raw Yte:", Yte.shape, Yte.dtype)

    Xtr = array_to_2ch(Xtr, name="train_inputs.npy")
    Ytr = array_to_2ch(Ytr, name="train_labels.npy")
    Xte = array_to_2ch(Xte, name="test_inputs.npy")
    Yte = array_to_2ch(Yte, name="test_labels.npy")

    Xtr = torch.from_numpy(Xtr).float()
    Ytr = torch.from_numpy(Ytr).float()
    Xte = torch.from_numpy(Xte).float()
    Yte = torch.from_numpy(Yte).float()

    expected_x_shape = (2, 64, 32 * cfg.history_tti)
    expected_y_shape = (2, 64, 32 * cfg.pred_tti)

    assert Xtr.shape[1:] == expected_x_shape, \
        f"train input shape error: {Xtr.shape[1:]}, expected {expected_x_shape}"

    assert Ytr.shape[1:] == expected_y_shape, \
        f"train label shape error: {Ytr.shape[1:]}, expected {expected_y_shape}"

    assert Xte.shape[1:] == expected_x_shape, \
        f"test input shape error: {Xte.shape[1:]}, expected {expected_x_shape}"

    assert Yte.shape[1:] == expected_y_shape, \
        f"test label shape error: {Yte.shape[1:]}, expected {expected_y_shape}"

    return (Xtr, Ytr), (Xte, Yte)


def make_loaders(train_pair, test_pair, use_cuda):
    Xtr, Ytr = train_pair
    Xte, Yte = test_pair

    train_ds = TensorDataset(Xtr, Ytr)
    test_ds = TensorDataset(Xte, Yte)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    return train_loader, test_loader


def make_amp_things(amp_enabled: bool, device: torch.device):
    amp_enabled = bool(amp_enabled and device.type == "cuda")

    try:
        from torch.cuda.amp import GradScaler as _GradScaler
        scaler = _GradScaler(enabled=amp_enabled)
    except Exception:
        from torch.amp import GradScaler as _GradScaler
        scaler = _GradScaler(enabled=amp_enabled)

    try:
        from torch.cuda.amp import autocast as _autocast_old

        def autocast_ctx(enabled=True):
            return _autocast_old(enabled=bool(enabled and device.type == "cuda"))

    except Exception:
        from torch.amp import autocast as _autocast_new

        def autocast_ctx(enabled=True):
            return _autocast_new(
                device_type=device.type,
                enabled=bool(enabled and device.type == "cuda")
            )

    return scaler, autocast_ctx


@torch.no_grad()
def evaluate(model, loader, device, criterion, amp_enabled=True):
    model.eval()

    loss_sum = 0.0
    n = 0

    _, autocast_ctx = make_amp_things(amp_enabled, device)

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        with autocast_ctx(enabled=amp_enabled):
            pred = model(xb)
            loss = criterion(pred, yb)

        bs = xb.size(0)
        loss_sum += loss.item() * bs
        n += bs

    mean_nmse = loss_sum / max(1, n)
    mean_nmse_db = to_db(mean_nmse)

    return mean_nmse, mean_nmse_db


def main():
    assert cfg.epochs > cfg.warmup_epochs, "epochs 必须大于 warmup_epochs"
    assert cfg.history_tti >= 1, "history_tti 必须大于等于 1"
    assert cfg.pred_tti >= 1, "pred_tti 必须大于等于 1"

    set_seed(cfg.seed)

    run_name = f"8to{cfg.pred_tti}_{timestamp()}"
    run_dir = os.path.join(cfg.out_root, run_name)
    ensure_dir(run_dir)

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    log_path = os.path.join(run_dir, "training.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("")

    train_pair, test_pair = load_data(cfg.data_dir)

    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"

    train_loader, test_loader = make_loaders(
        train_pair,
        test_pair,
        use_cuda
    )

    model = TransformerRecon(
        in_ch=2,
        H=64,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        norm_first=cfg.norm_first,
    ).to(device)

    total_params, trainable_params = count_parameters(model)

    header = [
        f"Device: {device}",
        f"Model Parameters: total={total_params:,}, trainable={trainable_params:,}",
        f"Config: {asdict(cfg)}",
    ]

    print("\n".join(header))

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")

    criterion = NMSELoss(eps=1e-12, reduction="mean")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    warmup = LinearLR(
        optimizer,
        start_factor=0.2,
        total_iters=cfg.warmup_epochs
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=cfg.eta_min
    )

    scheduler = SequentialLR(
        optimizer,
        [warmup, cosine],
        milestones=[cfg.warmup_epochs]
    )

    scaler, autocast_ctx = make_amp_things(cfg.amp, device)

    best = float("inf")
    best_epoch = 0
    best_db = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            ncols=110
        )

        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx(enabled=cfg.amp):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    cfg.grad_clip
                )

            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            train_loss_sum += loss.item() * bs
            seen += bs

            pbar.set_postfix(
                nmse=f"{loss.item():.4e}",
                nmse_db=f"{to_db(loss.item()):.2f} dB",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )

        train_nmse = train_loss_sum / max(1, seen)
        train_nmse_db = to_db(train_nmse)

        test_nmse, test_nmse_db = evaluate(
            model,
            test_loader,
            device,
            criterion,
            amp_enabled=cfg.amp
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        line = (
            f"epoch: {epoch}, "
            f"train_nmse: {train_nmse:.6e} ({train_nmse_db:.2f} dB), "
            f"test_nmse: {test_nmse:.6e} ({test_nmse_db:.2f} dB), "
            f"lr: {lr_now:.6g}"
        )

        print(line)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        if epoch % cfg.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "train_nmse": train_nmse,
                    "train_nmse_db": train_nmse_db,
                    "test_nmse": test_nmse,
                    "test_nmse_db": test_nmse_db,
                    "lr": lr_now,
                    "config": asdict(cfg),
                },
                os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
            )

        if test_nmse < best - 1e-12:
            best = test_nmse
            best_db = test_nmse_db
            best_epoch = epoch

            torch.save(
                {
                    "model": model.state_dict(),
                    "best_test_nmse": best,
                    "best_test_nmse_db": best_db,
                    "epoch": epoch,
                    "config": asdict(cfg),
                },
                os.path.join(run_dir, "best.pt")
            )

    torch.save(
        {
            "model": model.state_dict(),
            "final_test_nmse": test_nmse,
            "final_test_nmse_db": test_nmse_db,
            "best_test_nmse": best,
            "best_test_nmse_db": best_db,
            "best_epoch": best_epoch,
            "epoch": cfg.epochs,
            "config": asdict(cfg),
        },
        os.path.join(run_dir, "last.pt")
    )

    print("\n训练完成。")
    print(f"输出目录：{run_dir}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best test NMSE: {best:.6e}")
    print(f"Best test NMSE dB: {best_db:.2f} dB")


if __name__ == "__main__":
    main()
