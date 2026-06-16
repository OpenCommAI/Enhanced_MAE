import os
import math
import numpy as np
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =========================
# RNN 模型：必须与训练时一致
# =========================
class RNNRecon(nn.Module):
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
        hidden_size=512,
        num_layers=2,
        bidirectional=True,
        dropout=0.1,
        rnn_nonlinearity="tanh",
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

        self.rnn = nn.RNN(
            input_size=self.in_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=rnn_nonlinearity,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        feat_out = hidden_size * (2 if bidirectional else 1)

        self.reduce_len = nn.AdaptiveAvgPool1d(self.W_out)

        self.head = nn.Sequential(
            nn.LayerNorm(feat_out),
            nn.Linear(feat_out, in_ch * H),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (C, H, W) == (self.in_ch, self.H, self.W_in), \
            f"expect (B,{self.in_ch},{self.H},{self.W_in}), got {tuple(x.shape)}"

        seq = x.permute(0, 3, 1, 2).contiguous().view(B, self.W_in, C * H)
        seq_out, _ = self.rnn(seq)

        seq_out = seq_out.transpose(1, 2).contiguous()
        seq_out = self.reduce_len(seq_out)
        seq_out = seq_out.transpose(1, 2).contiguous()

        step_feat = self.head(seq_out)
        step_feat = step_feat.view(B, self.W_out, self.in_ch, self.H)
        step_feat = step_feat.permute(0, 2, 3, 1).contiguous()
        return step_feat


# =========================
# 配置
# =========================
@dataclass
class Config:
    data_dir: str = r"/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/multi_model_all_zhibiao/CSI_data_25dB/None_svd_speed_test_8to1_10to100_step2_v1/all_speed_dataset"
    ckpt_path: str = r"/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/multi_model_all_zhibiao/Pth_all_model/RNN.pt"

    history_tti: int = 8
    pred_tti: int = 1

    batch_size: int = 128
    num_workers: int = 0
    device: str = "cuda:2" if torch.cuda.is_available() else "cpu"

    hidden_size: int = 512
    num_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.1
    rnn_nonlinearity: str = "tanh"

    eps: float = 1e-12
    save_per_sample: bool = True

    # RTX 3090 FP32 theoretical throughput
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
    Xte = np.load(os.path.join(data_dir, "test_inputs.npy"))
    Yte = np.load(os.path.join(data_dir, "test_labels.npy"))

    print("Raw Xte:", Xte.shape, Xte.dtype)
    print("Raw Yte:", Yte.shape, Yte.dtype)

    if np.iscomplexobj(Xte):
        Xte = complex_to_2ch(Xte)
        Yte = complex_to_2ch(Yte)

    Xte = torch.from_numpy(Xte).float()
    Yte = torch.from_numpy(Yte).float()

    expected_x_shape = (2, 64, 32 * history_tti)
    expected_y_shape = (2, 64, 32 * pred_tti)

    assert Xte.shape[1:] == expected_x_shape, \
        f"test input shape error: {Xte.shape[1:]}, expected {expected_x_shape}"
    assert Yte.shape[1:] == expected_y_shape, \
        f"test label shape error: {Yte.shape[1:]}, expected {expected_y_shape}"

    return Xte, Yte


def make_loader(X, Y, batch_size, num_workers, use_cuda):
    ds = TensorDataset(X, Y)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
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
    """
    Time = FLOPs / (gpu_tflops * 1e12)
    """
    if flops is None:
        return float("nan")
    return flops / (gpu_tflops * 1e12)


# =========================
# FLOPs
# =========================
def compute_flops(model, device, history_tti):
    """
    计算单样本前向 FLOPs
    依赖 thop:
        pip install thop
    """
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
def evaluate_metrics(model, loader, device, eps=1e-12, save_per_sample=True):
    model.eval()

    sse_global = 0.0
    power_global = 0.0

    total_elems = 0
    sum_y = 0.0
    sum_y2 = 0.0

    nmse_each = []
    evm_each = []
    r2_each = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

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

    print("Config:", asdict(cfg))
    print("Device:", device)
    if use_cuda:
        print("Actual CUDA device:", torch.cuda.get_device_name(device))

    # 加载测试集
    Xte, Yte = load_test_data(cfg.data_dir, cfg.history_tti, cfg.pred_tti)
    loader = make_loader(Xte, Yte, cfg.batch_size, cfg.num_workers, use_cuda)

    # 构建模型
    model = RNNRecon(
        in_ch=2,
        H=64,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
        rnn_nonlinearity=cfg.rnn_nonlinearity,
    ).to(device)

    # 加载权重
    ckpt = load_checkpoint(model, cfg.ckpt_path, device)
    if isinstance(ckpt, dict) and "config" in ckpt:
        print("Checkpoint config:", ckpt["config"])

    # 测试集指标
    results = evaluate_metrics(
        model=model,
        loader=loader,
        device=device,
        eps=cfg.eps,
        save_per_sample=cfg.save_per_sample,
    )

    # Params
    total_params, trainable_params = count_parameters(model)
    ckpt_size_mb = get_ckpt_size_mb(cfg.ckpt_path)

    # FLOPs
    flops = compute_flops(model, device, cfg.history_tti)

    # Theoretical time on RTX 3090
    theory_time = theoretical_time_from_flops(flops, cfg.gpu_tflops_fp32) if flops is not None else float("nan")

    # 输出
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

    # 保存逐样本指标
    if cfg.save_per_sample:
        np.save(os.path.join(cfg.data_dir, "nmse_per_sample.npy"), results["nmse_per_sample"])
        np.save(os.path.join(cfg.data_dir, "r2_per_sample.npy"), results["r2_per_sample"])
        np.save(os.path.join(cfg.data_dir, "evm_per_sample.npy"), results["evm_per_sample"])

        valid_nmse = results["nmse_per_sample"][~np.isnan(results["nmse_per_sample"])]
        valid_r2 = results["r2_per_sample"][~np.isnan(results["r2_per_sample"])]
        valid_evm = results["evm_per_sample"][~np.isnan(results["evm_per_sample"])]

        if valid_nmse.size > 0:
            print(f"[NMSE]   per-sample mean={valid_nmse.mean():.6e}, std={valid_nmse.std():.6e}")
        if valid_r2.size > 0:
            print(f"[R2]     per-sample mean={valid_r2.mean():.6f}, std={valid_r2.std():.6f}, "
                  f"NaN count={results['r2_per_sample'].size - valid_r2.size}")
        if valid_evm.size > 0:
            print(f"[EVM]    per-sample mean={valid_evm.mean():.4f}%, std={valid_evm.std():.4f}%, "
                  f"NaN count={results['evm_per_sample'].size - valid_evm.size}")

    # 保存 summary
    summary_path = os.path.join(cfg.data_dir, "rnn_eval_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Config: {asdict(cfg)}\n")
        f.write(f"Device: {device}\n")
        if use_cuda:
            f.write(f"Actual CUDA device: {torch.cuda.get_device_name(device)}\n")
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

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
