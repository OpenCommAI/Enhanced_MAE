import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# 位置编码（正余弦）
# =========================
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        t = x.size(1)
        return x + self.pe[:t].unsqueeze(0)


# =========================
# Transformer 模型：与训练代码一致
# =========================
class TransformerRecon(nn.Module):
    """
    输入 : (B, 2, 64, 32*history_tti)
    输出 : (B, 2, 64, 32*pred_tti)
    """
    def __init__(
        self,
        in_ch=2,
        H=64,
        history_tti=8,
        pred_tti=1,
        d_model=256,
        nhead=8,
        num_layers=5,
        dim_feedforward=1024,
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
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len=self.W_in)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

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
        b, c, h, w = x.shape
        assert (c, h, w) == (self.in_ch, self.H, self.W_in), \
            f"expect (B,{self.in_ch},{self.H},{self.W_in}), got {tuple(x.shape)}"

        seq = x.permute(0, 3, 1, 2).contiguous().view(b, self.W_in, c * h)

        seq = self.input_proj(seq)
        seq = self.pos_encoding(seq)
        seq_out = self.encoder(seq)

        seq_out = seq_out.transpose(1, 2).contiguous()
        seq_out = self.reduce_len(seq_out)
        seq_out = seq_out.transpose(1, 2).contiguous()

        step_feat = self.head(seq_out)
        step_feat = step_feat.view(b, self.W_out, self.in_ch, self.H)
        pred = step_feat.permute(0, 2, 3, 1).contiguous()
        return pred


# =========================
# 配置
# =========================
@dataclass
class Config:
    data_dir: str = r"./data/paper_metrics/None_svd_speed_test_8to1_10to100_step2_v1/all_speed_dataset"
    ckpt_path: str = r"./checkpoints/Transformer_last.pt"
    out_dir: str = r"./transformer_eval_outputs"

    history_tti: int = 8
    pred_tti: int = 1

    batch_size: int = 128
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    amp: bool = True

    d_model: int = 256
    nhead: int = 8
    num_layers: int = 5
    dim_feedforward: int = 768
    dropout: float = 0.1
    norm_first: bool = True

    eps: float = 1e-12
    save_per_sample: bool = True

    gpu_name_for_theory: str = "NVIDIA RTX 3090"
    gpu_tflops_fp32: float = 35.58


cfg = Config()


# =========================
# 工具函数
# =========================
def to_db(x: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(x), eps))


def format_count(num: float) -> str:
    if num >= 1e12:
        return f"{num / 1e12:.2f}T"
    if num >= 1e9:
        return f"{num / 1e9:.2f}G"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    if num >= 1e3:
        return f"{num / 1e3:.2f}K"
    return str(num)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_device(device_cfg: str):
    if device_cfg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device_cfg}', but CUDA is unavailable.")
        if device_cfg == "cuda":
            return torch.device("cuda")
        idx = int(device_cfg.split(":")[1])
        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device '{device_cfg}', but only {torch.cuda.device_count()} CUDA device(s) are available."
            )
        return torch.device(device_cfg)

    if device_cfg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device config: {device_cfg}")


def complex_to_2ch(x: np.ndarray):
    """
    输入:
      x: (N, 64, 32*k), complex
    输出:
      x: (N, 2, 64, 32*k), float32
    """
    assert x.ndim == 3, f"expect 3D array, got {x.ndim}D"
    real = np.real(x)
    imag = np.imag(x)
    x = np.stack([real, imag], axis=1)
    return x.astype(np.float32)


def load_test_data(data_dir, history_tti, pred_tti):
    xte = np.load(os.path.join(data_dir, "test_inputs.npy"))
    yte = np.load(os.path.join(data_dir, "test_labels.npy"))

    print("Raw Xte:", xte.shape, xte.dtype)
    print("Raw Yte:", yte.shape, yte.dtype)

    if np.iscomplexobj(xte):
        xte = complex_to_2ch(xte)
        yte = complex_to_2ch(yte)

    xte = torch.from_numpy(xte).float()
    yte = torch.from_numpy(yte).float()

    expected_x_shape = (2, 64, 32 * history_tti)
    expected_y_shape = (2, 64, 32 * pred_tti)

    assert tuple(xte.shape[1:]) == expected_x_shape, \
        f"test input shape error: {xte.shape[1:]}, expected {expected_x_shape}"
    assert tuple(yte.shape[1:]) == expected_y_shape, \
        f"test label shape error: {yte.shape[1:]}, expected {expected_y_shape}"

    return xte, yte


def make_loader(x, y, batch_size, num_workers, use_cuda):
    ds = TensorDataset(x, y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


def load_checkpoint(model: nn.Module, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[warning] load_state_dict not exact match")
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
    return ckpt


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_ckpt_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def theoretical_time_from_flops(flops: float, gpu_tflops: float) -> float:
    if flops is None:
        return float("nan")
    return flops / (gpu_tflops * 1e12)


def make_autocast_ctx(use_cuda: bool, amp_enabled: bool):
    amp_enabled = bool(use_cuda and amp_enabled)

    try:
        def autocast_ctx():
            if not amp_enabled:
                return nullcontext()
            return torch.amp.autocast(device_type="cuda")
    except AttributeError:
        from torch.cuda.amp import autocast

        def autocast_ctx():
            if not amp_enabled:
                return nullcontext()
            return autocast()

    return autocast_ctx


# =========================
# FLOPs
# =========================
def compute_flops(model, device, history_tti):
    try:
        from thop import profile
    except ImportError:
        print("[warning] thop is not installed. Please run: pip install thop")
        return None

    model.eval()
    dummy = torch.randn(1, 2, 64, 32 * history_tti, device=device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    return flops


# =========================
# 指标计算
# =========================
@torch.no_grad()
def evaluate_metrics(model, loader, device, eps=1e-12, amp_enabled=True, save_per_sample=True):
    model.eval()
    use_cuda = device.type == "cuda"
    autocast_ctx = make_autocast_ctx(use_cuda, amp_enabled)

    sse_global = 0.0
    power_global = 0.0

    total_elems = 0
    sum_y = 0.0
    sum_y2 = 0.0

    nmse_each = []
    evm_each = []
    r2_each = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=use_cuda)
        yb = yb.to(device, non_blocking=use_cuda)

        with autocast_ctx():
            yhat = model(xb)

        y = yb.reshape(yb.size(0), -1)
        ph = yhat.reshape(yhat.size(0), -1)

        err = y - ph
        sse = torch.sum(err ** 2, dim=1)
        power = torch.sum(y ** 2, dim=1)

        sse_global += sse.sum().item()
        power_global += power.sum().item()

        sum_y += torch.sum(y).item()
        sum_y2 += torch.sum(y ** 2).item()
        total_elems += y.numel()

        if save_per_sample:
            nmse = sse / torch.clamp(power, min=eps)
            evm = torch.sqrt(sse / torch.clamp(power, min=eps)) * 100.0

            y_mean = y.mean(dim=1, keepdim=True)
            sst = torch.sum((y - y_mean) ** 2, dim=1)
            r2 = torch.where(
                sst > eps,
                1.0 - sse / sst,
                torch.tensor(float("nan"), device=sst.device),
            )

            nmse_each.append(nmse.cpu())
            evm_each.append(evm.cpu())
            r2_each.append(r2.cpu())

    global_nmse = float("nan") if power_global <= eps else (sse_global / power_global)
    global_nmse_db = to_db(global_nmse)
    global_evm = float("nan") if power_global <= eps else (math.sqrt(sse_global / power_global) * 100.0)

    y_mean_global = sum_y / max(1, total_elems)
    sst_global = sum_y2 - total_elems * (y_mean_global ** 2)
    global_r2 = float("nan") if sst_global <= eps else (1.0 - sse_global / sst_global)

    results = {
        "global_nmse": global_nmse,
        "global_nmse_db": global_nmse_db,
        "global_evm_percent": global_evm,
        "global_r2": global_r2,
    }

    if save_per_sample:
        results["nmse_per_sample"] = torch.cat(nmse_each, dim=0).numpy()
        results["evm_per_sample"] = torch.cat(evm_each, dim=0).numpy()
        results["r2_per_sample"] = torch.cat(r2_each, dim=0).numpy()

    return results


# =========================
# 主函数
# =========================
def main():
    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"
    ensure_dir(cfg.out_dir)

    print("Eval config:", asdict(cfg))
    print("Device:", device)
    if use_cuda:
        print("Actual CUDA device:", torch.cuda.get_device_name(device))

    ckpt_probe = torch.load(cfg.ckpt_path, map_location="cpu")
    ckpt_cfg = ckpt_probe.get("config", {}) if isinstance(ckpt_probe, dict) else {}

    if ckpt_cfg:
        print("Checkpoint config:", ckpt_cfg)

    history_tti = ckpt_cfg.get("history_tti", cfg.history_tti)
    pred_tti = ckpt_cfg.get("pred_tti", cfg.pred_tti)
    d_model = ckpt_cfg.get("d_model", cfg.d_model)
    nhead = ckpt_cfg.get("nhead", cfg.nhead)
    num_layers = ckpt_cfg.get("num_layers", cfg.num_layers)
    dim_feedforward = ckpt_cfg.get("dim_feedforward", cfg.dim_feedforward)
    dropout = ckpt_cfg.get("dropout", cfg.dropout)
    norm_first = ckpt_cfg.get("norm_first", cfg.norm_first)

    xte, yte = load_test_data(cfg.data_dir, history_tti, pred_tti)
    loader = make_loader(xte, yte, cfg.batch_size, cfg.num_workers, use_cuda)

    model = TransformerRecon(
        in_ch=2,
        H=64,
        history_tti=history_tti,
        pred_tti=pred_tti,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        norm_first=norm_first,
    ).to(device)

    ckpt = load_checkpoint(model, cfg.ckpt_path, device)

    results = evaluate_metrics(
        model=model,
        loader=loader,
        device=device,
        eps=cfg.eps,
        amp_enabled=cfg.amp,
        save_per_sample=cfg.save_per_sample,
    )

    total_params, trainable_params = count_parameters(model)
    ckpt_size_mb = get_ckpt_size_mb(cfg.ckpt_path)
    flops = compute_flops(model, device, history_tti)
    theory_time = theoretical_time_from_flops(flops, cfg.gpu_tflops_fp32) if flops is not None else float("nan")

    print()
    print(f"[NMSE]   test global: {results['global_nmse']:.6e} ({results['global_nmse_db']:.2f} dB)")
    print(f"[R2]     test global: {results['global_r2']:.6f}")
    print(f"[EVM]    test global: {results['global_evm_percent']:.4f}%")
    print(f"[Params] total: {total_params} ({format_count(total_params)})")
    print(f"[Params] trainable: {trainable_params} ({format_count(trainable_params)})")
    print(f"[Size]   checkpoint: {ckpt_size_mb:.2f} MB")
    if flops is not None:
        print(f"[FLOPs]  single-sample forward: {flops:.0f} ({format_count(flops)})")
        print(f"[Time]   theoretical on {cfg.gpu_name_for_theory}: {theory_time:.6e} s")

    if isinstance(ckpt, dict):
        for key in ["epoch", "best_test_nmse", "best_test_nmse_db", "test_nmse", "test_nmse_db", "lr"]:
            if key in ckpt:
                print(f"[CKPT]   {key}: {ckpt[key]}")

    if cfg.save_per_sample:
        np.save(os.path.join(cfg.out_dir, "nmse_per_sample.npy"), results["nmse_per_sample"])
        np.save(os.path.join(cfg.out_dir, "r2_per_sample.npy"), results["r2_per_sample"])
        np.save(os.path.join(cfg.out_dir, "evm_per_sample.npy"), results["evm_per_sample"])

        valid_nmse = results["nmse_per_sample"][~np.isnan(results["nmse_per_sample"])]
        valid_r2 = results["r2_per_sample"][~np.isnan(results["r2_per_sample"])]
        valid_evm = results["evm_per_sample"][~np.isnan(results["evm_per_sample"])]

        if valid_nmse.size > 0:
            print(f"[NMSE]   per-sample mean={valid_nmse.mean():.6e}, std={valid_nmse.std():.6e}")
        if valid_r2.size > 0:
            print(
                f"[R2]     per-sample mean={valid_r2.mean():.6f}, std={valid_r2.std():.6f}, "
                f"NaN count={results['r2_per_sample'].size - valid_r2.size}"
            )
        if valid_evm.size > 0:
            print(
                f"[EVM]    per-sample mean={valid_evm.mean():.4f}%, std={valid_evm.std():.4f}%, "
                f"NaN count={results['evm_per_sample'].size - valid_evm.size}"
            )

    save_json(os.path.join(cfg.out_dir, "eval_config.json"), asdict(cfg))

    summary_path = os.path.join(cfg.out_dir, "transformer_eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Eval config: {asdict(cfg)}\n")
        if ckpt_cfg:
            f.write(f"Checkpoint config: {ckpt_cfg}\n")
        f.write(f"Device: {device}\n")
        if use_cuda:
            f.write(f"Actual CUDA device: {torch.cuda.get_device_name(device)}\n")
        f.write(f"history_tti: {history_tti}\n")
        f.write(f"pred_tti: {pred_tti}\n")
        f.write(f"d_model: {d_model}\n")
        f.write(f"nhead: {nhead}\n")
        f.write(f"num_layers: {num_layers}\n")
        f.write(f"dim_feedforward: {dim_feedforward}\n")
        f.write(f"dropout: {dropout}\n")
        f.write(f"norm_first: {norm_first}\n")
        f.write(f"Test NMSE: {results['global_nmse']:.6e}\n")
        f.write(f"Test NMSE(dB): {results['global_nmse_db']:.2f}\n")
        f.write(f"Test R2: {results['global_r2']:.6f}\n")
        f.write(f"Test EVM(%): {results['global_evm_percent']:.4f}\n")
        f.write(f"Params total: {total_params} ({format_count(total_params)})\n")
        f.write(f"Params trainable: {trainable_params} ({format_count(trainable_params)})\n")
        f.write(f"Checkpoint size(MB): {ckpt_size_mb:.2f}\n")
        if flops is not None:
            f.write(f"FLOPs: {flops:.0f} ({format_count(flops)})\n")
            f.write(f"Theoretical Time on {cfg.gpu_name_for_theory}: {theory_time:.6e} s\n")
        if isinstance(ckpt, dict):
            for key in ["epoch", "best_test_nmse", "best_test_nmse_db", "test_nmse", "test_nmse_db", "lr"]:
                if key in ckpt:
                    f.write(f"{key}: {ckpt[key]}\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()