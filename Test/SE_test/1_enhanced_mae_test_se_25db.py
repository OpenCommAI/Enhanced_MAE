import os
import csv
import math
import random
import numpy as np
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_mae_multi_hop_random_v1 import build_mae_channel_target_hop3


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class Config:
    # -------------------------
    # 数据配置
    # -------------------------
    num2 = 20
    num1 = 90

    data_root: str = fr"/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/multi_E_multi_dB/multi_e_CSI_data/"
    all_dataset_name: str = fr"{num2}dB_svd0{num1}_norm_8to1_dataset"

    history_tti: int = 8
    pred_tti: int = 1

    # -------------------------
    # 测试配置
    # -------------------------
    batch_size: int = 128
    num_workers: int = 1

    amp: bool = True
    seed: int = 42
    device: str = "cuda:3"

    # -------------------------
    # 跳频配置，要和训练时保持一致
    # -------------------------
    hop_steps: Tuple[int, ...] = (1, 3, 5, 7)
    balanced_hop: bool = True
    eval_all_hops: bool = True


cfg = Config()


# ============================================================
# 2. Paper-style SE 测试配置
# ============================================================

SE_SNR_DB = 20
num3 = 90

CKPT_PATH = "/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/multi_model_SE_test/result_0605_multi_SE/outputs_enhanced_mae_20dB_svd090_8to1/enhanced_mae_8to1_2026-06-05_11-26-45/best.pt"

SAVE_CSV = fr"./result_0605_multi_SE/test_paper_style_SE_{SE_SNR_DB}dB_denorm_powernorm_results.csv"

# 这里改成你保存 4 个 Z-score npy 文件的目录
ZSCORE_STATS_DIR = fr"./Zscore_original/{SE_SNR_DB}dB_svd0{num3}_norm"

MEAN_REAL_NAME = fr"global_mean_real_after_svd0{num3}.npy"
MEAN_IMAG_NAME = fr"global_mean_imag_after_svd0{num3}.npy"
STD_REAL_NAME = fr"global_std_real_after_svd0{num3}.npy"
STD_IMAG_NAME = fr"global_std_imag_after_svd0{num3}.npy"

# # 这里改成你保存 4 个 Z-score npy 文件的目录
# ZSCORE_STATS_DIR = fr"./Zscore_original/25dB_zscore_norm"

# MEAN_REAL_NAME = fr"global_mean_real_raw.npy"
# MEAN_IMAG_NAME = fr"global_mean_imag_raw.npy"
# STD_REAL_NAME = fr"global_std_real_raw.npy"
# STD_IMAG_NAME = fr"global_std_imag_raw.npy"

# # 改成你的 Z-score 参数目录
# ZSCORE_STATS_DIR = "./Zscore_original/25dB_zscore_norm"

# # 如果你的文件名还是 after_svd090 这一组，代码会自动兼容候选文件名
# MEAN_REAL_CANDIDATES = [
#     "global_mean_real_raw.npy"
# ]

# MEAN_IMAG_CANDIDATES = [
#     "global_mean_imag_raw.npy"
# ]

# STD_REAL_CANDIDATES = [
#     "global_std_real_raw.npy"
# ]

# STD_IMAG_CANDIDATES = [
#     "global_std_imag_raw.npy"
# ]

# 你的 CSI 维度:
#   (B, 2, 64, 32 * pred_tti)
#
# 含义:
#   64 = 子载波维度
#   32 = 天线维度
NUM_SUBCARRIERS = 64
NUM_ANTENNAS = 32

# mean: 对 pred_tti 和 64 个子载波取平均
# sum : 对 64 个子载波求和，对 pred_tti 取平均
SE_REDUCTION = "mean"

# 是否在反标准化后做信道功率归一化
# 这里建议保持 True
DO_CHANNEL_POWER_NORM = True


# ============================================================
# 3. 基础工具函数
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str):
    if device_cfg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA 不可用，自动切换到 CPU。")
            return torch.device("cpu")

        if device_cfg == "cuda":
            return torch.device("cuda")

        idx = int(device_cfg.split(":")[1])

        if idx >= torch.cuda.device_count():
            raise RuntimeError(
                f"Requested device '{device_cfg}', "
                f"but only {torch.cuda.device_count()} CUDA device(s) are available."
            )

        return torch.device(device_cfg)

    if device_cfg == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device config: {device_cfg}")


def make_amp_things(use_cuda: bool, amp_enabled: bool):
    amp_enabled = bool(use_cuda and amp_enabled)

    def autocast_ctx():
        if not amp_enabled:
            return nullcontext()

        try:
            return torch.amp.autocast(device_type="cuda", enabled=True)
        except Exception:
            from torch.cuda.amp import autocast
            return autocast()

    return autocast_ctx


def nmse_to_db(nmse_value: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(nmse_value), eps))


# ============================================================
# 4. 数据加载函数
# ============================================================

def complex_to_2ch(x: np.ndarray):
    assert x.ndim == 3, f"expect 3D complex array, got {x.ndim}D, shape={x.shape}"

    real = np.real(x)
    imag = np.imag(x)

    x = np.stack([real, imag], axis=1)

    return x.astype(np.float32)


def array_to_2ch(x: np.ndarray, name: str):
    """
    统一转成:
        (N, 2, 64, W)

    支持:
        complex 3D: (N, 64, W)
        real 3D   : (N, 64, W)
        4D        : (N, 2, 64, W)
        4D        : (N, 64, W, 2)
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


def load_pair(inputs_path: str, labels_path: str):
    X = np.load(inputs_path)
    Y = np.load(labels_path)

    print(f"Raw inputs : {X.shape}, {X.dtype}")
    print(f"Raw labels : {Y.shape}, {Y.dtype}")

    X = array_to_2ch(X, name=os.path.basename(inputs_path))
    Y = array_to_2ch(Y, name=os.path.basename(labels_path))

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    print(f"Processed inputs : {X.shape}, {X.dtype}")
    print(f"Processed labels : {Y.shape}, {Y.dtype}")

    return X, Y


def load_test_dataset_only():
    """
    只加载测试集:
        test_inputs.npy
        test_labels.npy
    """
    all_dir = os.path.join(cfg.data_root, cfg.all_dataset_name)

    test_inputs_path = os.path.join(all_dir, "test_inputs.npy")
    test_labels_path = os.path.join(all_dir, "test_labels.npy")

    for path in [test_inputs_path, test_labels_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到测试数据文件: {path}")

    print("\n" + "=" * 100)
    print("加载测试集")
    print(f"All dataset dir : {all_dir}")
    print(f"Test inputs     : {test_inputs_path}")
    print(f"Test labels     : {test_labels_path}")
    print("=" * 100)

    Xte, Yte = load_pair(test_inputs_path, test_labels_path)

    return Xte, Yte


class TTIDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, history_tti: int, pred_tti: int):
        self.X = X
        self.Y = Y

        expected_x_shape = (2, 64, 32 * history_tti)
        expected_y_shape = (2, 64, 32 * pred_tti)

        assert self.X.shape[1:] == expected_x_shape, (
            f"X shape wrong: got {self.X.shape[1:]}, expected {expected_x_shape}"
        )

        assert self.Y.shape[1:] == expected_y_shape, (
            f"Y shape wrong: got {self.Y.shape[1:]}, expected {expected_y_shape}"
        )

        assert len(self.X) == len(self.Y), (
            f"X 和 Y 样本数量不一致: len(X)={len(self.X)}, len(Y)={len(self.Y)}"
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_test_loader(Xte, Yte, use_cuda: bool):
    dataset = TTIDataset(
        Xte,
        Yte,
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    return loader, len(dataset)


# ============================================================
# 5. Z-score 参数加载与反标准化
# ============================================================

def load_zscore_stats(stats_dir: str):
    """
    加载 4 个 Z-score 参数文件:
        global_mean_real_after_svd090.npy
        global_mean_imag_after_svd090.npy
        global_std_real_after_svd090.npy
        global_std_imag_after_svd090.npy
    """

    paths = {
        "mean_real": os.path.join(stats_dir, MEAN_REAL_NAME),
        "mean_imag": os.path.join(stats_dir, MEAN_IMAG_NAME),
        "std_real": os.path.join(stats_dir, STD_REAL_NAME),
        "std_imag": os.path.join(stats_dir, STD_IMAG_NAME),
    }

    for key, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到 Z-score 参数文件 {key}: {path}")

    stats = {
        "mean_real": np.load(paths["mean_real"]),
        "mean_imag": np.load(paths["mean_imag"]),
        "std_real": np.load(paths["std_real"]),
        "std_imag": np.load(paths["std_imag"]),
    }

    print("\n" + "=" * 100)
    print("加载 Z-score 反标准化参数")
    print(f"Stats dir: {stats_dir}")

    for key, value in stats.items():
        print(
            f"{key:10s}: shape={value.shape}, dtype={value.dtype}, "
            f"min={np.min(value):.6e}, max={np.max(value):.6e}, "
            f"mean={np.mean(value):.6e}, abs_mean={np.mean(np.abs(value)):.6e}"
        )

    print("=" * 100)

    return stats


def _prepare_stat_tensor(stat_np, target_real_tensor: torch.Tensor, name: str):
    """
    把 mean/std 的 numpy 数组整理成可以和:
        target_real_tensor: (B, 64, W)
    广播的 torch tensor。

    兼容:
        scalar
        (64, 32)
        (1, 64, 32)
        (64, 1)
        (1, 32)
        (64,)
        (32,)
    """

    device = target_real_tensor.device
    dtype = target_real_tensor.dtype

    B, K, W = target_real_tensor.shape

    stat = torch.as_tensor(stat_np, device=device, dtype=dtype)
    stat = stat.squeeze()

    if stat.dim() == 0:
        return stat

    if stat.dim() == 1:
        if stat.numel() == K:
            return stat.view(1, K, 1)

        if stat.numel() == W:
            return stat.view(1, 1, W)

        if stat.numel() == NUM_ANTENNAS and W % NUM_ANTENNAS == 0:
            repeat_t = W // NUM_ANTENNAS
            stat = stat.view(1, 1, NUM_ANTENNAS).repeat(1, 1, repeat_t)
            return stat

        raise ValueError(
            f"{name} 是 1D，但长度 {stat.numel()} 无法匹配目标 shape={(B, K, W)}"
        )

    if stat.dim() == 2:
        h, w = stat.shape

        if h == K and w == W:
            return stat.view(1, K, W)

        if h == K and w == NUM_ANTENNAS and W % NUM_ANTENNAS == 0:
            repeat_t = W // NUM_ANTENNAS
            stat = stat.view(1, K, NUM_ANTENNAS).repeat(1, 1, repeat_t)
            return stat

        if h == K and w == 1:
            return stat.view(1, K, 1)

        if h == 1 and w == W:
            return stat.view(1, 1, W)

        if h == 1 and w == NUM_ANTENNAS and W % NUM_ANTENNAS == 0:
            repeat_t = W // NUM_ANTENNAS
            stat = stat.view(1, 1, NUM_ANTENNAS).repeat(1, 1, repeat_t)
            return stat

        raise ValueError(
            f"{name} 是 2D，shape={tuple(stat.shape)} 无法匹配目标 shape={(B, K, W)}"
        )

    if stat.dim() == 3:
        if stat.shape[-2:] == (K, W):
            return stat.reshape(1, K, W)

        if stat.shape[-2:] == (K, NUM_ANTENNAS) and W % NUM_ANTENNAS == 0:
            repeat_t = W // NUM_ANTENNAS
            stat = stat.reshape(1, K, NUM_ANTENNAS).repeat(1, 1, repeat_t)
            return stat

        try:
            torch.broadcast_shapes(stat.shape, target_real_tensor.shape)
            return stat
        except Exception:
            pass

        raise ValueError(
            f"{name} 是 3D，shape={tuple(stat.shape)} 无法匹配目标 shape={(B, K, W)}"
        )

    raise ValueError(f"{name} 维度过多: shape={tuple(stat.shape)}")


@torch.no_grad()
def inverse_zscore_2ch(x_norm_2ch: torch.Tensor, stats: dict):
    """
    对 2-channel CSI 反 Z-score。

    输入:
        x_norm_2ch: (B, 2, 64, 32 * pred_tti)

    反标准化:
        real_raw = real_norm * std_real + mean_real
        imag_raw = imag_norm * std_imag + mean_imag

    输出:
        x_raw_2ch: (B, 2, 64, 32 * pred_tti)
    """

    assert x_norm_2ch.dim() == 4, f"x_norm_2ch 维度错误: {x_norm_2ch.shape}"
    assert x_norm_2ch.shape[1] == 2, f"通道维应该为 2，实际是 {x_norm_2ch.shape[1]}"
    assert x_norm_2ch.shape[2] == NUM_SUBCARRIERS, (
        f"子载波维应该为 {NUM_SUBCARRIERS}，实际是 {x_norm_2ch.shape[2]}"
    )

    real_norm = x_norm_2ch[:, 0, :, :].float()
    imag_norm = x_norm_2ch[:, 1, :, :].float()

    mean_real = _prepare_stat_tensor(stats["mean_real"], real_norm, "mean_real")
    mean_imag = _prepare_stat_tensor(stats["mean_imag"], imag_norm, "mean_imag")
    std_real = _prepare_stat_tensor(stats["std_real"], real_norm, "std_real")
    std_imag = _prepare_stat_tensor(stats["std_imag"], imag_norm, "std_imag")

    real_raw = real_norm * std_real + mean_real
    imag_raw = imag_norm * std_imag + mean_imag

    x_raw_2ch = torch.stack([real_raw, imag_raw], dim=1)

    return x_raw_2ch


# ============================================================
# 6. 模型构建和 checkpoint 加载
# ============================================================

def build_model():
    model = build_mae_channel_target_hop3(
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        hop_steps=cfg.hop_steps,
        balanced_hop=cfg.balanced_hop,
    )

    return model


def load_checkpoint(model, ckpt_path: str, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print(f"成功加载 checkpoint 中的 ['model'] 权重: {ckpt_path}")
    else:
        model.load_state_dict(ckpt)
        print(f"成功加载纯 state_dict 权重: {ckpt_path}")

    model.to(device)
    model.eval()

    return model


# ============================================================
# 7. NMSE、功率统计、信道功率归一化
# ============================================================

def nmse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12):
    """
    pred/target:
        (B, 2, 64, 32 * pred_tti)
    """
    diff = (pred - target).reshape(pred.size(0), -1)
    tgt = target.reshape(target.size(0), -1)

    nmse = diff.pow(2).sum(dim=1) / (tgt.pow(2).sum(dim=1) + eps)
    nmse_mean = nmse.mean()

    return nmse_mean


@torch.no_grad()
def csi_power_mean(x_2ch: torch.Tensor):
    """
    平均元素功率:
        mean(real^2 + imag^2)
    """
    real = x_2ch[:, 0].float()
    imag = x_2ch[:, 1].float()

    power = real.pow(2) + imag.pow(2)

    return power.mean().item()


@torch.no_grad()
def csi_power_sum_count(x_2ch: torch.Tensor):
    """
    返回元素功率总和与元素数量，用于全测试集估计平均功率。
    """
    real = x_2ch[:, 0].float()
    imag = x_2ch[:, 1].float()

    power = real.pow(2) + imag.pow(2)

    return power.sum().item(), power.numel()


@torch.no_grad()
def estimate_global_true_raw_power(test_loader, device, zscore_stats):
    """
    先扫一遍测试集，只用真实标签 y_true_norm。
    反标准化后统计 raw CSI 的全局平均元素功率。

    后续统一用:
        scale = sqrt(global_true_raw_power)

    做:
        H_scaled = H_raw / scale

    这样保证测试集真实 raw CSI 的平均元素功率约等于 1。
    """

    use_cuda = device.type == "cuda"

    power_sum = 0.0
    power_count = 0

    pbar = tqdm(
        test_loader,
        desc="Estimating global true raw CSI power",
        ncols=120,
    )

    for _, y_true_norm in pbar:
        yb_norm = y_true_norm.to(device, non_blocking=use_cuda)
        yb_raw = inverse_zscore_2ch(yb_norm, zscore_stats)

        ps, pc = csi_power_sum_count(yb_raw)
        power_sum += ps
        power_count += pc

    global_power = power_sum / max(power_count, 1)
    power_scale = math.sqrt(max(global_power, 1e-30))

    print("\n" + "=" * 100)
    print("测试集真实 raw CSI 全局功率估计")
    print(f"global_true_raw_element_power : {global_power:.6e}")
    print(f"power_scale = sqrt(power)     : {power_scale:.6e}")
    print("后续会执行: H_scaled = H_raw / power_scale")
    print("归一化后，真实 CSI 平均元素功率应约等于 1")
    print("=" * 100)

    return global_power, power_scale


@torch.no_grad()
def apply_channel_power_norm(x_raw_2ch: torch.Tensor, power_scale: float):
    """
    反标准化后的 raw CSI 再做信道功率归一化:
        H_scaled = H_raw / sqrt(E[|H_true_raw|^2])
    """
    if not DO_CHANNEL_POWER_NORM:
        return x_raw_2ch

    return x_raw_2ch / float(power_scale)


# ============================================================
# 8. 论文式 SE 计算函数
# ============================================================

@torch.no_grad()
def split_2ch_to_complex_tti(x_2ch: torch.Tensor):
    """
    输入:
        x_2ch: (B, 2, 64, 32 * pred_tti)

    维度含义:
        64 = 子载波 / RB 数
        32 = 天线数

    输出:
        H: (B, pred_tti, 64, 32)
           即 (B, T, K, Nt)
    """

    assert x_2ch.dim() == 4, f"x_2ch 维度错误: {x_2ch.shape}"
    assert x_2ch.shape[1] == 2, f"通道维应该为 2，实际是 {x_2ch.shape[1]}"
    assert x_2ch.shape[2] == NUM_SUBCARRIERS, (
        f"子载波维应该为 {NUM_SUBCARRIERS}，实际是 {x_2ch.shape[2]}"
    )

    B, _, K, W = x_2ch.shape

    assert W % NUM_ANTENNAS == 0, (
        f"W={W} 不能被天线数 {NUM_ANTENNAS} 整除，无法按 TTI 切分"
    )

    pred_tti = W // NUM_ANTENNAS

    real = x_2ch[:, 0, :, :].float()
    imag = x_2ch[:, 1, :, :].float()

    H_complex = torch.complex(real, imag)

    H_complex = H_complex.reshape(B, K, pred_tti, NUM_ANTENNAS)
    H_complex = H_complex.permute(0, 2, 1, 3).contiguous()

    return H_complex


@torch.no_grad()
def calc_paper_style_se(
    pred_2ch: torch.Tensor,
    true_2ch: torch.Tensor,
    snr_db: float = 25.0,
    reduction: str = "mean",
    eps: float = 1e-12,
):
    """
    按论文方式计算 SE。

    注意:
        这里输入应当是:
            反标准化 + 信道功率归一化 后的 CSI。

    数据维度:
        (B, 2, 64, 32 * pred_tti)

    解释:
        64 = 子载波
        32 = 天线

    论文式逻辑:
        w_hat_k = h_hat_k / ||h_hat_k||
        SE_k = log2(1 + SNR * |h_true_k^H w_hat_k|^2)
    """

    assert reduction in ["mean", "sum"], "reduction 必须是 'mean' 或 'sum'"

    H_hat = split_2ch_to_complex_tti(pred_2ch)
    H_true = split_2ch_to_complex_tti(true_2ch)

    # H_hat/H_true: (B, T, K=64, Nt=32)

    snr_linear = 10.0 ** (snr_db / 10.0)

    # 预测 CSI 归一化得到 MRT / matched-filtering 预编码
    norm = torch.sqrt(
        torch.sum(torch.abs(H_hat) ** 2, dim=3, keepdim=True) + eps
    )

    W_hat = H_hat / norm

    # 真实 CSI 下的等效增益 h_true^H w_hat
    effective_gain = torch.sum(
        torch.conj(H_true) * W_hat,
        dim=3,
    )

    se_per_subcarrier = torch.log2(
        1.0 + snr_linear * torch.abs(effective_gain) ** 2
    )

    if reduction == "mean":
        se_per_sample = se_per_subcarrier.mean(dim=(1, 2))
    else:
        se_per_sample = se_per_subcarrier.sum(dim=2).mean(dim=1)

    return se_per_sample


# ============================================================
# 9. 测试集 Paper-style SE 评估
# ============================================================

@torch.no_grad()
def evaluate_test_paper_style_se_25db(
    model,
    test_loader,
    device,
    zscore_stats,
    power_scale: float,
):
    model.eval()

    use_cuda = device.type == "cuda"
    autocast_ctx = make_amp_things(use_cuda, cfg.amp)

    if cfg.eval_all_hops:
        eval_hops = cfg.hop_steps
    else:
        eval_hops = [None]

    pred_se_sum = 0.0
    pred_se_count = 0

    perfect_se_sum = 0.0
    perfect_se_count = 0

    nmse_norm_sum = 0.0
    nmse_raw_sum = 0.0
    nmse_count = 0

    pred_raw_power_sum = 0.0
    true_raw_power_sum = 0.0

    pred_scaled_power_sum = 0.0
    true_scaled_power_sum = 0.0

    power_count = 0

    csv_rows = []

    pbar = tqdm(
        test_loader,
        desc=f"Testing denorm + powernorm paper-style SE at {SE_SNR_DB} dB",
        ncols=170,
    )

    for batch_idx, (x_input, y_true_norm) in enumerate(pbar):
        xb = x_input.to(device, non_blocking=use_cuda)
        yb_norm = y_true_norm.to(device, non_blocking=use_cuda)

        # 真实 CSI 反标准化
        yb_raw = inverse_zscore_2ch(yb_norm, zscore_stats)

        # 真实 CSI 信道功率归一化
        yb_scaled = apply_channel_power_norm(yb_raw, power_scale)

        # Perfect CSI SE:
        # 用真实 scaled CSI 自己生成预编码，再用真实 scaled CSI 计算 SE
        se_perfect = calc_paper_style_se(
            pred_2ch=yb_scaled,
            true_2ch=yb_scaled,
            snr_db=SE_SNR_DB,
            reduction=SE_REDUCTION,
        )

        perfect_se_batch_mean = se_perfect.mean().item()
        perfect_se_sum += se_perfect.sum().item()
        perfect_se_count += se_perfect.numel()

        true_raw_power_batch = csi_power_mean(yb_raw)
        true_scaled_power_batch = csi_power_mean(yb_scaled)

        for hop in eval_hops:
            with autocast_ctx():
                if hop is None:
                    pred_norm = model(xb)
                    hop_value = -1
                else:
                    pred_norm = model(
                        xb,
                        y_label=None,
                        force_hop_step=int(hop),
                    )
                    hop_value = int(hop)

            # 预测 CSI 反标准化
            pred_raw = inverse_zscore_2ch(pred_norm, zscore_stats)

            # 预测 CSI 信道功率归一化
            pred_scaled = apply_channel_power_norm(pred_raw, power_scale)

            # 用预测 scaled CSI 生成预编码，用真实 scaled CSI 计算 SE
            se_pred = calc_paper_style_se(
                pred_2ch=pred_scaled,
                true_2ch=yb_scaled,
                snr_db=SE_SNR_DB,
                reduction=SE_REDUCTION,
            )

            pred_se_batch_mean = se_pred.mean().item()
            pred_se_sum += se_pred.sum().item()
            pred_se_count += se_pred.numel()

            # NMSE:
            # 标准化域 NMSE 与训练 loss 对齐
            # raw NMSE 用反标准化后的 CSI 计算
            # scaled NMSE 与 raw NMSE 数值相同，因为 pred/true 同除一个全局 scale
            loss_norm = nmse_loss(pred_norm, yb_norm)
            loss_raw = nmse_loss(pred_raw, yb_raw)

            bs = xb.size(0)
            nmse_norm_sum += loss_norm.item() * bs
            nmse_raw_sum += loss_raw.item() * bs
            nmse_count += bs

            pred_raw_power_batch = csi_power_mean(pred_raw)
            pred_scaled_power_batch = csi_power_mean(pred_scaled)

            pred_raw_power_sum += pred_raw_power_batch
            true_raw_power_sum += true_raw_power_batch

            pred_scaled_power_sum += pred_scaled_power_batch
            true_scaled_power_sum += true_scaled_power_batch

            power_count += 1

            se_gap_batch = perfect_se_batch_mean - pred_se_batch_mean
            se_ratio_batch = pred_se_batch_mean / max(perfect_se_batch_mean, 1e-12)

            raw_power_ratio_batch = pred_raw_power_batch / max(true_raw_power_batch, 1e-12)
            scaled_power_ratio_batch = pred_scaled_power_batch / max(true_scaled_power_batch, 1e-12)

            csv_rows.append([
                batch_idx,
                hop_value,
                pred_se_batch_mean,
                perfect_se_batch_mean,
                se_gap_batch,
                se_ratio_batch,
                loss_norm.item(),
                nmse_to_db(loss_norm.item()),
                loss_raw.item(),
                nmse_to_db(loss_raw.item()),
                pred_raw_power_batch,
                true_raw_power_batch,
                raw_power_ratio_batch,
                pred_scaled_power_batch,
                true_scaled_power_batch,
                scaled_power_ratio_batch,
            ])

            pbar.set_postfix(
                pred_SE=f"{pred_se_batch_mean:.4f}",
                perfect_SE=f"{perfect_se_batch_mean:.4f}",
                ratio=f"{se_ratio_batch:.4f}",
                norm_nmse_db=f"{nmse_to_db(loss_norm.item()):.2f}",
                raw_nmse_db=f"{nmse_to_db(loss_raw.item()):.2f}",
                pwr_scaled=f"{true_scaled_power_batch:.3f}",
            )

    pred_se_mean = pred_se_sum / max(pred_se_count, 1)
    perfect_se_mean = perfect_se_sum / max(perfect_se_count, 1)

    nmse_norm_mean = nmse_norm_sum / max(nmse_count, 1)
    nmse_raw_mean = nmse_raw_sum / max(nmse_count, 1)

    nmse_norm_db = nmse_to_db(nmse_norm_mean)
    nmse_raw_db = nmse_to_db(nmse_raw_mean)

    se_gap = perfect_se_mean - pred_se_mean
    se_ratio = pred_se_mean / max(perfect_se_mean, 1e-12)

    pred_raw_power_mean = pred_raw_power_sum / max(power_count, 1)
    true_raw_power_mean = true_raw_power_sum / max(power_count, 1)

    pred_scaled_power_mean = pred_scaled_power_sum / max(power_count, 1)
    true_scaled_power_mean = true_scaled_power_sum / max(power_count, 1)

    raw_power_ratio = pred_raw_power_mean / max(true_raw_power_mean, 1e-12)
    scaled_power_ratio = pred_scaled_power_mean / max(true_scaled_power_mean, 1e-12)

    result = {
        "pred_se_mean": pred_se_mean,
        "perfect_se_mean": perfect_se_mean,
        "se_gap": se_gap,
        "se_ratio": se_ratio,

        "nmse_norm_mean": nmse_norm_mean,
        "nmse_norm_db": nmse_norm_db,
        "nmse_raw_mean": nmse_raw_mean,
        "nmse_raw_db": nmse_raw_db,

        "pred_raw_power_mean": pred_raw_power_mean,
        "true_raw_power_mean": true_raw_power_mean,
        "raw_power_ratio": raw_power_ratio,

        "pred_scaled_power_mean": pred_scaled_power_mean,
        "true_scaled_power_mean": true_scaled_power_mean,
        "scaled_power_ratio": scaled_power_ratio,

        "csv_rows": csv_rows,
    }

    return result


# ============================================================
# 10. 主函数
# ============================================================

def main():
    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"

    print("\n" + "=" * 100)
    print("25 dB 测试集 Paper-style SE 计算：反 Z-score + 信道功率归一化")
    print(f"Device                 : {device}")
    print(f"SNR                    : {SE_SNR_DB} dB")
    print(f"Checkpoint             : {CKPT_PATH}")
    print(f"Dataset                : {os.path.join(cfg.data_root, cfg.all_dataset_name)}")
    print(f"Zscore stats           : {ZSCORE_STATS_DIR}")
    print(f"history_tti            : {cfg.history_tti}")
    print(f"pred_tti               : {cfg.pred_tti}")
    print(f"hop_steps              : {cfg.hop_steps}")
    print(f"eval_all_hops          : {cfg.eval_all_hops}")
    print(f"NUM_SUBCARRIERS        : {NUM_SUBCARRIERS}")
    print(f"NUM_ANTENNAS           : {NUM_ANTENNAS}")
    print(f"SE_REDUCTION           : {SE_REDUCTION}")
    print(f"DO_CHANNEL_POWER_NORM  : {DO_CHANNEL_POWER_NORM}")
    print("=" * 100)

    # 1. 加载 Z-score 参数
    zscore_stats = load_zscore_stats(ZSCORE_STATS_DIR)

    # 2. 只加载测试集
    Xte, Yte = load_test_dataset_only()

    test_loader, test_size = make_test_loader(
        Xte,
        Yte,
        use_cuda=use_cuda,
    )

    print(f"Test samples: {test_size}")

    # 3. 估计测试集真实 raw CSI 的全局平均元素功率
    global_true_raw_power, power_scale = estimate_global_true_raw_power(
        test_loader=test_loader,
        device=device,
        zscore_stats=zscore_stats,
    )

    # 4. 构建模型
    model = build_model()

    # 5. 加载 checkpoint
    model = load_checkpoint(
        model=model,
        ckpt_path=CKPT_PATH,
        device=device,
    )

    # 6. 测试反标准化 + 功率归一化后的 paper-style SE
    result = evaluate_test_paper_style_se_25db(
        model=model,
        test_loader=test_loader,
        device=device,
        zscore_stats=zscore_stats,
        power_scale=power_scale,
    )

    # 7. 打印结果
    print("\n" + "=" * 100)
    print("25 dB 测试集 Paper-style SE 结果：反 Z-score + 信道功率归一化")
    print(f"global true raw element power    : {global_true_raw_power:.6e}")
    print(f"power scale sqrt(power)          : {power_scale:.6e}")
    print("-" * 100)
    print(f"预测 CSI 预编码 SE               : {result['pred_se_mean']:.6f} bit/s/Hz")
    print(f"Perfect CSI SE                   : {result['perfect_se_mean']:.6f} bit/s/Hz")
    print(f"SE gap, perfect - pred           : {result['se_gap']:.6f} bit/s/Hz")
    print(f"SE ratio, pred / perfect         : {result['se_ratio']:.6f}")
    print("-" * 100)
    print(f"标准化域 NMSE                    : {result['nmse_norm_mean']:.6e}")
    print(f"标准化域 NMSE(dB)                : {result['nmse_norm_db']:.2f} dB")
    print(f"反标准化 raw 域 NMSE             : {result['nmse_raw_mean']:.6e}")
    print(f"反标准化 raw 域 NMSE(dB)         : {result['nmse_raw_db']:.2f} dB")
    print("-" * 100)
    print(f"预测 raw CSI 平均功率            : {result['pred_raw_power_mean']:.6e}")
    print(f"真实 raw CSI 平均功率            : {result['true_raw_power_mean']:.6e}")
    print(f"raw 功率比 pred / true           : {result['raw_power_ratio']:.6f}")
    print("-" * 100)
    print(f"预测 scaled CSI 平均功率         : {result['pred_scaled_power_mean']:.6e}")
    print(f"真实 scaled CSI 平均功率         : {result['true_scaled_power_mean']:.6e}")
    print(f"scaled 功率比 pred / true        : {result['scaled_power_ratio']:.6f}")
    print("=" * 100)

    # 8. 保存 batch 级别结果
    save_dir = os.path.dirname(SAVE_CSV)
    if save_dir != "":
        os.makedirs(save_dir, exist_ok=True)

    with open(SAVE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        writer.writerow([
            "batch_idx",
            "hop",
            "pred_precoding_se_mean_denorm_powernorm",
            "perfect_csi_se_mean_denorm_powernorm",
            "se_gap_perfect_minus_pred",
            "se_ratio_pred_over_perfect",
            "nmse_norm",
            "nmse_norm_db",
            "nmse_raw",
            "nmse_raw_db",
            "pred_raw_power",
            "true_raw_power",
            "raw_power_ratio_pred_over_true",
            "pred_scaled_power",
            "true_scaled_power",
            "scaled_power_ratio_pred_over_true",
        ])

        for row in result["csv_rows"]:
            writer.writerow(row)

    print(f"详细结果已保存到: {SAVE_CSV}")


if __name__ == "__main__":
    main()