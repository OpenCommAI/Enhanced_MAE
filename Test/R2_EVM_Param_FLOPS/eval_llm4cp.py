import json
import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from LLM4CP_models import build_llm4cp_csi


# =========================
# FLOPs 包装器：模型测试时只输入 x
# =========================
class LLM4CPPredWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


# =========================
# 配置
# =========================
@dataclass
class Config:
    # 数据集文件夹，里面需要有 test_inputs.npy 和 test_labels.npy
    data_dir: str = r"/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/multi_model_all_zhibiao/CSI_data_25dB/None_svd_speed_test_8to1_10to100_step2_v1/all_speed_dataset"

    # 模型权重
    ckpt_path: str = r"/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/multi_model_all_zhibiao/Pth_all_model/LLM4CP_best.pt"

    # 输出结果文件夹
    out_dir: str = r"./result_LLM4CP_eval"

    history_tti: int = 8
    pred_tti: int = 1

    batch_size: int = 128
    num_workers: int = 0
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 是否使用 AMP 测试
    amp: bool = True

    eps: float = 1e-12
    save_per_sample: bool = True
    save_predictions: bool = False

    gpu_name_for_theory: str = "NVIDIA RTX 3090"
    gpu_tflops_fp32: float = 35.58

    # GPT2 本地路径
    # 如果你本地有 gpt2 文件夹，就写本地路径
    # 如果你想联网下载，改成 gpt_path=None, local_files_only=False
    gpt_path: Optional[str] = r"/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/gpt2"
    local_files_only: bool = True


cfg = Config()


# =========================
# 基础工具
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


def resolve_device(device_cfg: str):
    if device_cfg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[warning] CUDA 不可用，自动使用 CPU")
            return torch.device("cpu")

        if device_cfg == "cuda":
            return torch.device("cuda")

        idx = int(device_cfg.split(":")[1])
        if idx >= torch.cuda.device_count():
            print(
                f"[warning] 指定 {device_cfg}，但只有 {torch.cuda.device_count()} 张 GPU，自动使用 cuda:0"
            )
            return torch.device("cuda:0")

        return torch.device(device_cfg)

    if device_cfg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device config: {device_cfg}")


def find_existing_file(data_dir: str, names):
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"在 {data_dir} 中没有找到这些文件：{names}")


def complex_to_2ch(x: np.ndarray):
    """
    complex:
        输入: (N, 64, 32*T)
        输出: (N, 2, 64, 32*T)
    """
    if x.ndim != 3:
        raise ValueError(f"complex 数据应为 3D，即 (N,64,W)，当前 shape={x.shape}")

    real = np.real(x)
    imag = np.imag(x)
    x = np.stack([real, imag], axis=1)
    return x.astype(np.float32)


def normalize_data_shape(arr: np.ndarray, name: str, expected_width: int):
    """
    支持三种格式：
    1. complex: (N, 64, W)
    2. 实虚双通道: (N, 2, 64, W)
    3. 实虚最后一维: (N, 64, W, 2)
    """
    print(f"Raw {name}: shape={arr.shape}, dtype={arr.dtype}")

    if np.iscomplexobj(arr):
        arr = complex_to_2ch(arr)

    elif arr.ndim == 4:
        if arr.shape[1] == 2:
            arr = arr.astype(np.float32)
        elif arr.shape[-1] == 2:
            arr = np.transpose(arr, (0, 3, 1, 2)).astype(np.float32)
        else:
            raise ValueError(
                f"{name} 是 4D，但不是 (N,2,64,W) 或 (N,64,W,2)，当前 shape={arr.shape}"
            )

    elif arr.ndim == 3:
        print(f"[warning] {name} 是实数 3D 数据，将自动补零虚部通道")
        arr = np.stack([arr, np.zeros_like(arr)], axis=1).astype(np.float32)

    else:
        raise ValueError(f"{name} 维度错误，当前 shape={arr.shape}")

    if arr.shape[1] != 2:
        raise ValueError(f"{name} 通道数应为 2，当前 shape={arr.shape}")

    if arr.shape[2] != 64:
        raise ValueError(f"{name} 高度应为 64，当前 shape={arr.shape}")

    if arr.shape[3] != expected_width:
        raise ValueError(
            f"{name} 宽度错误，当前为 {arr.shape[3]}，应为 {expected_width}"
        )

    print(f"Used {name}: shape={arr.shape}, dtype={arr.dtype}")
    return arr


def load_test_data(data_dir: str, history_tti: int, pred_tti: int):
    x_path = find_existing_file(
        data_dir,
        ["test_inputs.npy", "test_input.npy", "X_test.npy", "test_X.npy"],
    )
    y_path = find_existing_file(
        data_dir,
        ["test_labels.npy", "test_label.npy", "Y_test.npy", "test_Y.npy"],
    )

    xte = np.load(x_path)
    yte = np.load(y_path)

    xte = normalize_data_shape(
        xte,
        name="Xte",
        expected_width=32 * history_tti,
    )

    yte = normalize_data_shape(
        yte,
        name="Yte",
        expected_width=32 * pred_tti,
    )

    xte = torch.from_numpy(xte).float()
    yte = torch.from_numpy(yte).float()

    return xte, yte


class TTIDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        if len(self.x) != len(self.y):
            raise ValueError(f"x 和 y 样本数不一致: {len(self.x)} vs {len(self.y)}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def make_loader(x, y, batch_size, num_workers, use_cuda):
    dataset = TTIDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )


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
# checkpoint 与模型构建
# =========================
def load_checkpoint(ckpt_path: str, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    return torch.load(ckpt_path, map_location=device)


def parse_ckpt_config(ckpt) -> Dict[str, Any]:
    """
    兼容 checkpoint 里 config 是 dict / json 字符串 / dataclass 的情况。
    """
    if not isinstance(ckpt, dict):
        return {}

    ckpt_cfg = ckpt.get("config", {})

    if ckpt_cfg is None:
        return {}

    if isinstance(ckpt_cfg, dict):
        return ckpt_cfg

    if isinstance(ckpt_cfg, str):
        try:
            return json.loads(ckpt_cfg)
        except Exception:
            return {}

    if hasattr(ckpt_cfg, "__dict__"):
        return vars(ckpt_cfg)

    return {}


def build_model_kwargs(eval_cfg: Config, ckpt_cfg: dict):
    """
    重点：
    这里继续使用 build_llm4cp_csi，不改 import。

    但 build_llm4cp_csi() 内部已经固定传入了：
        num_subcarriers=64
        num_antennas=32
        in_chans=2
        proj_ant=4
        gpt_type="gpt2"
        gpt_layers=8
        d_model=512
        d_ff=512
        patch_size=2
        res_layers=2
        res_dim=64
        dropout=0.1
        freeze_gpt=True

    因此这里不能再传这些重复参数。
    这里只传 history_tti、pred_tti、gpt_path、local_files_only。
    """
    return {
        "history_tti": ckpt_cfg.get("history_tti", eval_cfg.history_tti),
        "pred_tti": ckpt_cfg.get("pred_tti", eval_cfg.pred_tti),
        "gpt_path": ckpt_cfg.get("gpt_path", eval_cfg.gpt_path),
        "local_files_only": ckpt_cfg.get("local_files_only", eval_cfg.local_files_only),
    }


def strip_prefix_if_present(state_dict, prefix: str):
    keys = list(state_dict.keys())

    if len(keys) == 0:
        return state_dict

    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def load_state_into_model(model: nn.Module, ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state = ckpt["model"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    state = strip_prefix_if_present(state, "module.")
    state = strip_prefix_if_present(state, "model.")

    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        print("[warning] load_state_dict 不是完全匹配")
        if missing:
            print("missing keys:", missing)
        if unexpected:
            print("unexpected keys:", unexpected)
    else:
        print("Checkpoint loaded successfully.")


# =========================
# 模型复杂度
# =========================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def param_memory_mb(num_params: int):
    """
    按 fp32 估算参数存储。
    """
    return num_params * 4 / (1024 ** 2)


def get_ckpt_size_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def theoretical_time_from_flops(flops: float, gpu_tflops: float) -> float:
    if flops is None:
        return float("nan")

    return flops / (gpu_tflops * 1e12)


def compute_flops(model, device, history_tti):
    try:
        from thop import profile
    except ImportError:
        print("[warning] thop 未安装，无法统计 FLOPs。安装命令：pip install thop")
        return None

    wrapper = LLM4CPPredWrapper(model).to(device)
    wrapper.eval()

    dummy_x = torch.randn(1, 2, 64, 32 * history_tti, device=device)

    flops, _ = profile(
        wrapper,
        inputs=(dummy_x,),
        verbose=False,
    )

    return flops


# =========================
# 指标计算
# =========================
@torch.no_grad()
def evaluate_metrics(
    model,
    loader,
    device,
    eps=1e-12,
    amp_enabled=True,
    save_per_sample=True,
    save_predictions=False,
):
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

    pred_all = []
    label_all = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=use_cuda)
        yb = yb.to(device, non_blocking=use_cuda)

        with autocast_ctx():
            pred = model(xb)

        pred = pred.float()
        yb = yb.float()

        y = yb.reshape(yb.size(0), -1)
        ph = pred.reshape(pred.size(0), -1)

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
            evm = torch.sqrt(nmse) * 100.0

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

        if save_predictions:
            pred_all.append(pred.cpu())
            label_all.append(yb.cpu())

    global_nmse = float("nan") if power_global <= eps else (sse_global / power_global)
    global_nmse_db = to_db(global_nmse, eps)
    global_evm = float("nan") if power_global <= eps else math.sqrt(global_nmse) * 100.0

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

    if save_predictions:
        results["pred_all"] = torch.cat(pred_all, dim=0).numpy()
        results["label_all"] = torch.cat(label_all, dim=0).numpy()

    return results


# =========================
# 保存结果
# =========================
def save_results(
    cfg,
    results,
    model_kwargs,
    ckpt_cfg,
    total_params,
    trainable_params,
    param_mem_mb_value,
    ckpt_size_mb,
    flops,
    theory_time,
    device,
):
    ensure_dir(cfg.out_dir)

    if cfg.save_per_sample:
        np.save(os.path.join(cfg.out_dir, "nmse_per_sample.npy"), results["nmse_per_sample"])
        np.save(os.path.join(cfg.out_dir, "r2_per_sample.npy"), results["r2_per_sample"])
        np.save(os.path.join(cfg.out_dir, "evm_per_sample.npy"), results["evm_per_sample"])

    if cfg.save_predictions:
        np.save(os.path.join(cfg.out_dir, "predictions.npy"), results["pred_all"])
        np.save(os.path.join(cfg.out_dir, "labels.npy"), results["label_all"])

    summary_path = os.path.join(cfg.out_dir, "llm4cp_eval_summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========== LLM4CP Test Summary ==========\n")
        f.write(f"Config: {asdict(cfg)}\n")
        f.write(f"Model kwargs: {model_kwargs}\n")

        if ckpt_cfg:
            f.write(f"Checkpoint config: {ckpt_cfg}\n")

        f.write(f"Device: {device}\n")

        if device.type == "cuda":
            f.write(f"Actual CUDA device: {torch.cuda.get_device_name(device)}\n")

        f.write("\n")
        f.write("========== Metrics ==========\n")
        f.write(f"Test NMSE: {results['global_nmse']:.6e}\n")
        f.write(f"Test NMSE(dB): {results['global_nmse_db']:.2f}\n")
        f.write(f"Test R2: {results['global_r2']:.6f}\n")
        f.write(f"Test EVM(%): {results['global_evm_percent']:.4f}\n")

        f.write("\n")
        f.write("========== Complexity ==========\n")
        f.write(f"Params total: {total_params} ({format_count(total_params)})\n")
        f.write(f"Params trainable: {trainable_params} ({format_count(trainable_params)})\n")
        f.write(f"Parameter memory(MB, fp32): {param_mem_mb_value:.2f}\n")
        f.write(f"Checkpoint size(MB): {ckpt_size_mb:.2f}\n")

        if flops is not None:
            f.write(f"FLOPs: {flops:.0f} ({format_count(flops)})\n")
            f.write(f"Theoretical Time on {cfg.gpu_name_for_theory}: {theory_time:.6e} s\n")

        if cfg.save_per_sample:
            valid_nmse = results["nmse_per_sample"][~np.isnan(results["nmse_per_sample"])]
            valid_r2 = results["r2_per_sample"][~np.isnan(results["r2_per_sample"])]
            valid_evm = results["evm_per_sample"][~np.isnan(results["evm_per_sample"])]

            f.write("\n")
            f.write("========== Per-sample Statistics ==========\n")

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

    print(f"Summary saved to: {summary_path}")


# =========================
# 主函数
# =========================
def main():
    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"
    ensure_dir(cfg.out_dir)

    print("========== Config ==========")
    print(asdict(cfg))

    print()
    print("========== Device ==========")
    print("Device:", device)

    if use_cuda:
        print("Actual CUDA device:", torch.cuda.get_device_name(device))

    print()
    print("========== Load Checkpoint ==========")
    ckpt = load_checkpoint(cfg.ckpt_path, device)
    ckpt_cfg = parse_ckpt_config(ckpt)

    if ckpt_cfg:
        print("Checkpoint config:", ckpt_cfg)

    print()
    print("========== Build Model ==========")
    model_kwargs = build_model_kwargs(cfg, ckpt_cfg)
    model = build_llm4cp_csi(**model_kwargs).to(device)

    print(model.__class__.__name__)
    print("Model kwargs:", model_kwargs)

    print()
    print("========== Load State Dict ==========")
    load_state_into_model(model, ckpt)

    history_tti = model_kwargs["history_tti"]
    pred_tti = model_kwargs["pred_tti"]

    print()
    print("========== Load Data ==========")
    xte, yte = load_test_data(
        data_dir=cfg.data_dir,
        history_tti=history_tti,
        pred_tti=pred_tti,
    )

    loader = make_loader(
        xte,
        yte,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        use_cuda=use_cuda,
    )

    print()
    print("========== Evaluate ==========")
    results = evaluate_metrics(
        model=model,
        loader=loader,
        device=device,
        eps=cfg.eps,
        amp_enabled=cfg.amp,
        save_per_sample=cfg.save_per_sample,
        save_predictions=cfg.save_predictions,
    )

    total_params, trainable_params = count_parameters(model)
    param_mem = param_memory_mb(total_params)
    ckpt_size = get_ckpt_size_mb(cfg.ckpt_path)

    print()
    print("========== FLOPs ==========")
    flops = compute_flops(
        model=model,
        device=device,
        history_tti=history_tti,
    )

    theory_time = (
        theoretical_time_from_flops(flops, cfg.gpu_tflops_fp32)
        if flops is not None
        else float("nan")
    )

    print()
    print("========== Test Results ==========")
    print(f"[NMSE]   test global: {results['global_nmse']:.6e} ({results['global_nmse_db']:.2f} dB)")
    print(f"[R2]     test global: {results['global_r2']:.6f}")
    print(f"[EVM]    test global: {results['global_evm_percent']:.4f}%")
    print(f"[Params] total: {total_params} ({format_count(total_params)})")
    print(f"[Params] trainable: {trainable_params} ({format_count(trainable_params)})")
    print(f"[Memory] parameter memory(fp32): {param_mem:.2f} MB")
    print(f"[Size]   checkpoint: {ckpt_size:.2f} MB")

    if flops is not None:
        print(f"[FLOPs]  single-sample forward: {flops:.0f} ({format_count(flops)})")
        print(f"[Time]   theoretical on {cfg.gpu_name_for_theory}: {theory_time:.6e} s")

    if cfg.save_per_sample:
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

    save_results(
        cfg=cfg,
        results=results,
        model_kwargs=model_kwargs,
        ckpt_cfg=ckpt_cfg,
        total_params=total_params,
        trainable_params=trainable_params,
        param_mem_mb_value=param_mem,
        ckpt_size_mb=ckpt_size,
        flops=flops,
        theory_time=theory_time,
        device=device,
    )


if __name__ == "__main__":
    main()