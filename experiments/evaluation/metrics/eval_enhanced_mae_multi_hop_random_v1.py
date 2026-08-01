import os
import math
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models_mae_multi_hop_random_v1 import build_mae_channel_target_hop3


# =========================
# 配置
# =========================
@dataclass
class Config:
    # 测试集目录，里面需要包含:
    # test_inputs.npy
    # test_labels.npy
    data_dir: str = r"./data/25dB_svd090_norm_8to1_dataset"

    # 模型权重路径
    ckpt_path: str = r"./checkpoints/best.pt"

    # 输出目录；如果为空，则默认保存到 data_dir
    output_dir: str = ""

    # X-to-1 中的 X
    history_tti: int = 8
    pred_tti: int = 1

    batch_size: int = 128
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # 模型跳频参数，需要与训练时一致
    hop_steps: Tuple[int, ...] = (1, 3, 5, 7)
    balanced_hop: bool = True

    # 测试时是否强制固定某一种 hop
    # None 表示按模型内部随机/均衡 hop 测试
    # 可以改成 1、3、5、7 做消融测试
    force_hop_step: Optional[int] = None

    # 随机种子。由于模型 forward 内部有随机 start row / hop，
    # 固定 seed 后，每次测试结果更容易复现
    seed: int = 2026

    eps: float = 1e-12
    save_per_sample: bool = True
    save_predictions: bool = False

    # RTX 3090 FP32 theoretical throughput
    gpu_name_for_theory: str = "NVIDIA RTX 3090"
    gpu_tflops_fp32: float = 35.58


cfg = Config()


# =========================
# 工具函数
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
                f"Requested device '{device_cfg}', but only "
                f"{torch.cuda.device_count()} CUDA device(s) are available."
            )

        return torch.device(device_cfg)

    if device_cfg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device config: {device_cfg}")


def make_torch_generator(device, seed: Optional[int]):
    if seed is None:
        return None

    try:
        if device.type == "cuda":
            gen = torch.Generator(device=device)
        else:
            gen = torch.Generator()
    except Exception:
        gen = torch.Generator()

    gen.manual_seed(seed)
    return gen


def complex_to_2ch(x: np.ndarray):
    """
    输入:
        x: (N, 64, 32*k), complex
    输出:
        x: (N, 2, 64, 32*k), float32
    """
    assert x.ndim == 3, f"expect 3D complex array, got {x.ndim}D"
    real = np.real(x)
    imag = np.imag(x)
    x = np.stack([real, imag], axis=1)
    return x.astype(np.float32)


def ensure_2ch_float32(x: np.ndarray, name: str):
    """
    支持两种输入格式:
    1. complex: (N, 64, 32*k)
    2. float:   (N, 2, 64, 32*k)
    """
    if np.iscomplexobj(x):
        return complex_to_2ch(x)

    if x.ndim == 4:
        return x.astype(np.float32)

    raise ValueError(
        f"{name} format error. Expected complex array with shape "
        f"(N,64,32*k) or float array with shape (N,2,64,32*k), "
        f"but got shape={x.shape}, dtype={x.dtype}"
    )


def load_test_data(data_dir, history_tti, pred_tti):
    x_path = os.path.join(data_dir, "test_inputs.npy")
    y_path = os.path.join(data_dir, "test_labels.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Cannot find {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Cannot find {y_path}")

    Xte = np.load(x_path)
    Yte = np.load(y_path)

    print("Raw Xte:", Xte.shape, Xte.dtype)
    print("Raw Yte:", Yte.shape, Yte.dtype)

    Xte = ensure_2ch_float32(Xte, "test_inputs.npy")
    Yte = ensure_2ch_float32(Yte, "test_labels.npy")

    print("2ch Xte:", Xte.shape, Xte.dtype)
    print("2ch Yte:", Yte.shape, Yte.dtype)

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

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    return loader


def clean_state_dict_keys(state_dict):
    """
    兼容 DataParallel / torch.compile 保存出来的 key。
    """
    new_state = {}

    for k, v in state_dict.items():
        new_k = k

        changed = True
        while changed:
            changed = False
            for prefix in ["module.", "_orig_mod."]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    changed = True

        new_state[new_k] = v

    return new_state


def extract_state_dict(ckpt):
    """
    兼容多种 checkpoint 保存格式:
    1. torch.save(model.state_dict())
    2. torch.save({"model": model.state_dict(), ...})
    3. torch.save({"state_dict": model.state_dict(), ...})
    4. torch.save({"model_state_dict": model.state_dict(), ...})
    """
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "net", "network"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]

        # 如果整个 ckpt 本身就是 state_dict
        if all(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt

    raise RuntimeError(
        "Cannot extract model state_dict from checkpoint. "
        "Please check your ckpt saving format."
    )


def load_checkpoint(model: nn.Module, ckpt_path: str, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = extract_state_dict(ckpt)
    state = clean_state_dict_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        print("[warning] load_state_dict not exact match")
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
    else:
        print("Checkpoint loaded successfully with exact key match.")

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


# =========================
# FLOPs 计算
# =========================
class MAEPredOnlyWrapper(nn.Module):
    """
    thop 只需要一个 tensor 输出，这里把 MAE 模型包装成只输出 pred。
    """
    def __init__(self, model, force_hop_step=None):
        super().__init__()
        self.model = model
        self.force_hop_step = force_hop_step

    def forward(self, x, y):
        loss, pred, mask, loss_dict = self.model(
            x,
            y,
            generator=None,
            force_hop_step=self.force_hop_step,
        )
        return pred


def compute_flops(model, device, history_tti, pred_tti, force_hop_step=None):
    """
    计算单样本前向 FLOPs。
    依赖 thop:
        pip install thop

    注意:
    该模型 forward 需要 x 和 y 两个输入，所以 dummy input 也需要两个。
    """
    try:
        from thop import profile
    except ImportError:
        print("[warning] thop is not installed. Please run: pip install thop")
        return None

    try:
        model.eval()

        wrapped_model = MAEPredOnlyWrapper(
            model=model,
            force_hop_step=force_hop_step,
        ).to(device)

        dummy_x = torch.randn(1, 2, 64, 32 * history_tti, device=device)
        dummy_y = torch.randn(1, 2, 64, 32 * pred_tti, device=device)

        flops, _ = profile(
            wrapped_model,
            inputs=(dummy_x, dummy_y),
            verbose=False,
        )

        return flops

    except Exception as e:
        print(f"[warning] FLOPs calculation failed: {repr(e)}")
        return None


# =========================
# 指标计算
# =========================
@torch.no_grad()
def evaluate_metrics(
    model,
    loader,
    device,
    eps=1e-12,
    save_per_sample=True,
    save_predictions=False,
    seed: Optional[int] = 2026,
    force_hop_step: Optional[int] = None,
):
    model.eval()

    generator = make_torch_generator(device, seed)

    sse_global = 0.0
    power_global = 0.0

    total_elems = 0
    sum_y = 0.0
    sum_y2 = 0.0

    nmse_each = []
    evm_each = []
    r2_each = []

    pred_each = []
    label_each = []

    hop_steps_all = []
    start_rows_all = []

    for batch_idx, (xb, yb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        # 关键区别：
        # 文档 1 的 MAE 模型 forward 需要 xb 和 yb，
        # 返回 loss, pred, mask, loss_dict
        loss, yhat, mask, loss_dict = model(
            xb,
            yb,
            generator=generator,
            force_hop_step=force_hop_step,
        )

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
                torch.full_like(sst, float("nan")),
            )

            nmse_each.append(nmse.detach().cpu())
            evm_each.append(evm.detach().cpu())
            r2_each.append(r2.detach().cpu())

        if save_predictions:
            pred_each.append(yhat.detach().cpu())
            label_each.append(yb.detach().cpu())

        if isinstance(loss_dict, dict):
            if "hop_steps" in loss_dict:
                hop_steps_all.append(loss_dict["hop_steps"].detach().cpu())
            if "start_rows" in loss_dict:
                start_rows_all.append(loss_dict["start_rows"].detach().cpu())

        if (batch_idx + 1) % 20 == 0:
            print(f"Evaluated {batch_idx + 1} batches...")

    global_nmse = float("nan") if power_global <= eps else (sse_global / power_global)
    global_nmse_db = to_db(global_nmse, eps=eps)

    global_evm = float("nan") if power_global <= eps else (
        math.sqrt(sse_global / power_global) * 100.0
    )

    y_mean_global = sum_y / max(1, total_elems)
    sst_global = sum_y2 - total_elems * (y_mean_global ** 2)
    global_r2 = float("nan") if sst_global <= eps else (
        1.0 - sse_global / sst_global
    )

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

    if save_predictions:
        results["predictions"] = torch.cat(pred_each, dim=0).numpy()
        results["labels"] = torch.cat(label_each, dim=0).numpy()

    if len(hop_steps_all) > 0:
        hop_steps = torch.cat(hop_steps_all, dim=0).numpy().astype(np.int64)
        unique_hop, counts_hop = np.unique(hop_steps, return_counts=True)
        results["hop_steps"] = hop_steps
        results["hop_distribution"] = {
            int(k): int(v) for k, v in zip(unique_hop, counts_hop)
        }

    if len(start_rows_all) > 0:
        start_rows = torch.cat(start_rows_all, dim=0).numpy().astype(np.int64)
        unique_start, counts_start = np.unique(start_rows, return_counts=True)
        results["start_rows"] = start_rows
        results["start_row_distribution"] = {
            int(k): int(v) for k, v in zip(unique_start, counts_start)
        }

    return results


# =========================
# 主函数
# =========================
def main():
    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"

    output_dir = cfg.output_dir if cfg.output_dir else cfg.data_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Config:", asdict(cfg))
    print("Device:", device)

    if use_cuda:
        print("Actual CUDA device:", torch.cuda.get_device_name(device))

    # 加载测试集
    Xte, Yte = load_test_data(
        data_dir=cfg.data_dir,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
    )

    loader = make_loader(
        X=Xte,
        Y=Yte,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_cuda=use_cuda,
    )

    # 构建模型
    model = build_mae_channel_target_hop3(
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        hop_steps=cfg.hop_steps,
        balanced_hop=cfg.balanced_hop,
    ).to(device)

    # 加载权重
    ckpt = load_checkpoint(
        model=model,
        ckpt_path=cfg.ckpt_path,
        device=device,
    )

    if isinstance(ckpt, dict) and "config" in ckpt:
        print("Checkpoint config:", ckpt["config"])

    # 测试集指标
    results = evaluate_metrics(
        model=model,
        loader=loader,
        device=device,
        eps=cfg.eps,
        save_per_sample=cfg.save_per_sample,
        save_predictions=cfg.save_predictions,
        seed=cfg.seed,
        force_hop_step=cfg.force_hop_step,
    )

    # Params
    total_params, trainable_params = count_parameters(model)

    # Checkpoint size
    ckpt_size_mb = get_ckpt_size_mb(cfg.ckpt_path)

    # FLOPs
    flops = compute_flops(
        model=model,
        device=device,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        force_hop_step=cfg.force_hop_step,
    )

    # Theoretical time on RTX 3090
    theory_time = theoretical_time_from_flops(
        flops=flops,
        gpu_tflops=cfg.gpu_tflops_fp32,
    ) if flops is not None else float("nan")

    # =========================
    # 输出结果
    # =========================
    print()
    print("========== Test Results ==========")
    print(f"[NMSE]   test global: {results['global_nmse']:.6e} ({results['global_nmse_db']:.2f} dB)")
    print(f"[R2]     test global: {results['global_r2']:.6f}")
    print(f"[EVM]    test global: {results['global_evm_percent']:.4f}%")
    print(f"[Params] total: {total_params} ({format_count(total_params)})")
    print(f"[Params] trainable: {trainable_params} ({format_count(trainable_params)})")
    print(f"[Size]   checkpoint: {ckpt_size_mb:.2f} MB")

    if flops is not None:
        print(f"[FLOPs]  single-sample forward: {flops:.0f} ({format_count(flops)})")
        print(f"[Time]   theoretical on {cfg.gpu_name_for_theory}: {theory_time:.6e} s")

    if "hop_distribution" in results:
        print(f"[Hop]    distribution: {results['hop_distribution']}")

    if "start_row_distribution" in results:
        print(f"[Start]  row distribution: {results['start_row_distribution']}")

    # =========================
    # 保存逐样本指标
    # =========================
    if cfg.save_per_sample:
        nmse_path = os.path.join(output_dir, "mae_nmse_per_sample.npy")
        r2_path = os.path.join(output_dir, "mae_r2_per_sample.npy")
        evm_path = os.path.join(output_dir, "mae_evm_per_sample.npy")

        np.save(nmse_path, results["nmse_per_sample"])
        np.save(r2_path, results["r2_per_sample"])
        np.save(evm_path, results["evm_per_sample"])

        valid_nmse = results["nmse_per_sample"][~np.isnan(results["nmse_per_sample"])]
        valid_r2 = results["r2_per_sample"][~np.isnan(results["r2_per_sample"])]
        valid_evm = results["evm_per_sample"][~np.isnan(results["evm_per_sample"])]

        if valid_nmse.size > 0:
            print(f"[NMSE]   per-sample mean={valid_nmse.mean():.6e}, std={valid_nmse.std():.6e}")

        if valid_r2.size > 0:
            print(
                f"[R2]     per-sample mean={valid_r2.mean():.6f}, "
                f"std={valid_r2.std():.6f}, "
                f"NaN count={results['r2_per_sample'].size - valid_r2.size}"
            )

        if valid_evm.size > 0:
            print(
                f"[EVM]    per-sample mean={valid_evm.mean():.4f}%, "
                f"std={valid_evm.std():.4f}%, "
                f"NaN count={results['evm_per_sample'].size - valid_evm.size}"
            )

        print(f"Per-sample NMSE saved to: {nmse_path}")
        print(f"Per-sample R2 saved to:   {r2_path}")
        print(f"Per-sample EVM saved to:  {evm_path}")

    # =========================
    # 保存预测结果，可选
    # =========================
    if cfg.save_predictions:
        pred_path = os.path.join(output_dir, "mae_predictions.npy")
        label_path = os.path.join(output_dir, "mae_labels.npy")

        np.save(pred_path, results["predictions"])
        np.save(label_path, results["labels"])

        print(f"Predictions saved to: {pred_path}")
        print(f"Labels saved to:      {label_path}")

    # =========================
    # 保存 hop / start row 信息
    # =========================
    if "hop_steps" in results:
        hop_path = os.path.join(output_dir, "mae_hop_steps.npy")
        np.save(hop_path, results["hop_steps"])
        print(f"Hop steps saved to:   {hop_path}")

    if "start_rows" in results:
        start_path = os.path.join(output_dir, "mae_start_rows.npy")
        np.save(start_path, results["start_rows"])
        print(f"Start rows saved to:  {start_path}")

    # =========================
    # 保存 summary
    # =========================
    summary_path = os.path.join(output_dir, "mae_multi_hop_random_eval_summary.txt")

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

        if "hop_distribution" in results:
            f.write(f"Hop distribution: {results['hop_distribution']}\n")

        if "start_row_distribution" in results:
            f.write(f"Start row distribution: {results['start_row_distribution']}\n")

        if cfg.save_per_sample:
            valid_nmse = results["nmse_per_sample"][~np.isnan(results["nmse_per_sample"])]
            valid_r2 = results["r2_per_sample"][~np.isnan(results["r2_per_sample"])]
            valid_evm = results["evm_per_sample"][~np.isnan(results["evm_per_sample"])]

            if valid_nmse.size > 0:
                f.write(f"Per-sample NMSE mean: {valid_nmse.mean():.6e}\n")
                f.write(f"Per-sample NMSE std: {valid_nmse.std():.6e}\n")

            if valid_r2.size > 0:
                f.write(f"Per-sample R2 mean: {valid_r2.mean():.6f}\n")
                f.write(f"Per-sample R2 std: {valid_r2.std():.6f}\n")
                f.write(f"Per-sample R2 NaN count: {results['r2_per_sample'].size - valid_r2.size}\n")

            if valid_evm.size > 0:
                f.write(f"Per-sample EVM mean(%): {valid_evm.mean():.4f}\n")
                f.write(f"Per-sample EVM std(%): {valid_evm.std():.4f}\n")
                f.write(f"Per-sample EVM NaN count: {results['evm_per_sample'].size - valid_evm.size}\n")

    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()