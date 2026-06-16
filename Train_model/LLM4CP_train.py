# llm4cp_train_no_speed_test_best_by_test_loss.py
import os
import math
import random
from contextlib import nullcontext
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from LLM4CP_models import build_llm4cp_csi


# ============================================================
# 1. 配置
# ============================================================
@dataclass
class Config:
    # ========================================================
    # 数据路径
    # ========================================================
    # 目录下应包含:
    #   all_speed_dataset/
    #       train_inputs.npy
    #       train_labels.npy
    #       test_inputs.npy
    #       test_labels.npy
    #
    # 注意：
    #   这里不再需要 speed_test_dataset/
    # ========================================================

    num1 = 8
    num2 = 25

    data_root: str = fr"/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/multi_E_multi_dB/multi_e_CSI_data/"

    all_dataset_name: str = fr"{num2}dB_zscore_norm_8to1_dataset"

    out_root: str = fr"./result_0603_multi_snr/outputs_llm4cp_{num2}dB_8to1"

    # ========================================================
    # 数据参数
    # ========================================================
    history_tti: int = 8
    pred_tti: int = 1

    # ========================================================
    # 训练参数
    # ========================================================
    batch_size: int = 128
    num_workers: int = 1
    epochs: int = 300

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    amp: bool = True
    seed: int = 42
    save_every: int = 50

    warmup_epochs: int = 10
    eta_min: float = 1e-6

    device: str = "cuda:0"

    # ========================================================
    # GPT-2 参数
    # ========================================================
    gpt_path: str = "/home/ubuntu/zq_mae/CSI_data_xtti2ytti_svdDone/gpt2"
    local_files_only: bool = True


cfg = Config()


# ============================================================
# 2. 工具函数
# ============================================================
def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


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

    try:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler(enabled=amp_enabled)
    except Exception:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    def autocast_ctx():
        if not amp_enabled:
            return nullcontext()

        try:
            from torch.cuda.amp import autocast
            return autocast()
        except Exception:
            return torch.amp.autocast(device_type="cuda")

    return scaler, autocast_ctx


def nmse_to_db(nmse_value: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(nmse_value), eps))


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


# ============================================================
# 3. 数据读取
# ============================================================
def complex_to_2ch(x: np.ndarray):
    """
    输入:
        x: (N, 64, 32*k), complex

    输出:
        x: (N, 2, 64, 32*k), float32
    """
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


def load_pair(inputs_path: str, labels_path: str):
    X = np.load(inputs_path)
    Y = np.load(labels_path)

    print(f"Raw inputs : {X.shape}, {X.dtype}")
    print(f"Raw labels : {Y.shape}, {Y.dtype}")

    X = array_to_2ch(X, name=os.path.basename(inputs_path))
    Y = array_to_2ch(Y, name=os.path.basename(labels_path))

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    expected_x_shape = (2, 64, 32 * cfg.history_tti)
    expected_y_shape = (2, 64, 32 * cfg.pred_tti)

    assert X.shape[1:] == expected_x_shape, (
        f"inputs shape wrong: got {X.shape[1:]}, expected {expected_x_shape}"
    )

    assert Y.shape[1:] == expected_y_shape, (
        f"labels shape wrong: got {Y.shape[1:]}, expected {expected_y_shape}"
    )

    print(f"Processed inputs : {X.shape}, {X.dtype}")
    print(f"Processed labels : {Y.shape}, {Y.dtype}")

    return X, Y


def load_all_dataset(data_root: str):
    all_dir = os.path.join(data_root, cfg.all_dataset_name)

    train_inputs_path = os.path.join(all_dir, "train_inputs.npy")
    train_labels_path = os.path.join(all_dir, "train_labels.npy")
    test_inputs_path = os.path.join(all_dir, "test_inputs.npy")
    test_labels_path = os.path.join(all_dir, "test_labels.npy")

    for path in [
        train_inputs_path,
        train_labels_path,
        test_inputs_path,
        test_labels_path,
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到数据文件: {path}")

    print("\n" + "=" * 100)
    print("加载总训练集和总测试集")
    print(f"All dataset dir: {all_dir}")
    print("=" * 100)

    Xtr, Ytr = load_pair(train_inputs_path, train_labels_path)
    Xte, Yte = load_pair(test_inputs_path, test_labels_path)

    return Xtr, Ytr, Xte, Yte


class TTIDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        assert len(self.X) == len(self.Y), (
            f"X 和 Y 样本数量不一致: len(X)={len(self.X)}, len(Y)={len(self.Y)}"
        )

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def make_loader(X, Y, use_cuda: bool, shuffle: bool):
    dataset = TTIDataset(X, Y)

    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    return loader, len(dataset)


def make_train_test_loaders(Xtr, Ytr, Xte, Yte, use_cuda: bool):
    train_loader, train_size = make_loader(
        Xtr,
        Ytr,
        use_cuda=use_cuda,
        shuffle=True,
    )

    test_loader, test_size = make_loader(
        Xte,
        Yte,
        use_cuda=use_cuda,
        shuffle=False,
    )

    return train_loader, test_loader, train_size, test_size


# ============================================================
# 4. 模型构建
# ============================================================
def build_model():
    model = build_llm4cp_csi(
        history_tti=cfg.history_tti,
        pred_tti=cfg.pred_tti,
        gpt_path=cfg.gpt_path,
        local_files_only=cfg.local_files_only,
    )

    return model


# ============================================================
# 5. 测试集评估
# ============================================================
@torch.no_grad()
def evaluate(model, loader, device, amp_enabled=True):
    model.eval()

    use_cuda = device.type == "cuda"
    _, autocast_ctx = make_amp_things(use_cuda, amp_enabled)

    loss_sum = 0.0
    n = 0

    for x_input, y_future in loader:
        xb = x_input.to(device, non_blocking=use_cuda)
        yb = y_future.to(device, non_blocking=use_cuda)

        with autocast_ctx():
            loss, pred, aux, loss_dict = model(xb, yb)

        bs = xb.size(0)
        loss_sum += loss.item() * bs
        n += bs

    nmse_mean = loss_sum / max(n, 1)
    nmse_db = nmse_to_db(nmse_mean)

    return nmse_mean, nmse_db


# ============================================================
# 6. checkpoint
# ============================================================
def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    epoch,
    train_loss,
    train_nmse_db,
    test_loss,
    test_nmse_db,
    best_epoch,
    best_test_loss,
    best_test_nmse_db,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,

            "epoch": epoch,

            "train_nmse": train_loss,
            "train_nmse_db": train_nmse_db,

            "test_nmse": test_loss,
            "test_nmse_db": test_nmse_db,

            "best_epoch": best_epoch,
            "best_test_nmse": best_test_loss,
            "best_test_nmse_db": best_test_nmse_db,

            "history_tti": cfg.history_tti,
            "pred_tti": cfg.pred_tti,
            "gpt_path": cfg.gpt_path,
            "config": asdict(cfg),
        },
        path,
    )


# ============================================================
# 7. 主函数
# ============================================================
def main():
    assert cfg.history_tti >= 1
    assert cfg.pred_tti >= 1
    assert cfg.epochs > cfg.warmup_epochs

    set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    use_cuda = device.type == "cuda"

    run_name = f"LLM4CP_train_8to{cfg.pred_tti}_{timestamp()}"
    run_dir = os.path.join(cfg.out_root, run_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)

    log_path = os.path.join(run_dir, "training.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(str(asdict(cfg)) + "\n")

    # ========================================================
    # 加载总训练集和总测试集
    # ========================================================
    Xtr, Ytr, Xte, Yte = load_all_dataset(cfg.data_root)

    train_loader, test_loader, train_size, test_size = make_train_test_loaders(
        Xtr,
        Ytr,
        Xte,
        Yte,
        use_cuda=use_cuda,
    )

    # ========================================================
    # 构建模型
    # ========================================================
    model = build_model().to(device)

    total_params, trainable_params = count_parameters(model)

    print("=" * 100)
    print("Model type: LLM4CP")
    print(f"Model Parameters: total={total_params:,}, trainable={trainable_params:,}")
    print(f"Device: {device}")
    print(f"Train samples: {train_size}")
    print(f"Test samples: {test_size}")
    print(f"Using local GPT-2 from: {cfg.gpt_path}")
    print(f"Config: {asdict(cfg)}")
    print("=" * 100)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write("Model type: LLM4CP\n")
        f.write(f"Model Parameters: total={total_params}, trainable={trainable_params}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Train samples: {train_size}\n")
        f.write(f"Test samples: {test_size}\n")
        f.write(f"Using local GPT-2 from: {cfg.gpt_path}\n")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.2,
        total_iters=cfg.warmup_epochs,
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs - cfg.warmup_epochs,
        eta_min=cfg.eta_min,
    )

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_epochs],
    )

    scaler, autocast_ctx = make_amp_things(use_cuda, cfg.amp)

    best_test_loss = float("inf")
    best_test_nmse_db = None
    best_epoch = 0

    last_train_loss = None
    last_train_nmse_db = None
    last_test_loss = None
    last_test_nmse_db = None

    best_path = os.path.join(run_dir, "best.pt")

    # ========================================================
    # 训练
    # ========================================================
    for epoch in range(1, cfg.epochs + 1):
        model.train()

        loss_running = 0.0
        seen = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{cfg.epochs}",
            ncols=120,
        )

        for x_input, y_future in pbar:
            xb = x_input.to(device, non_blocking=use_cuda)
            yb = y_future.to(device, non_blocking=use_cuda)

            optimizer.zero_grad(set_to_none=True)

            with autocast_ctx():
                loss, pred, aux, loss_dict = model(xb, yb)

            scaler.scale(loss).backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            bs = xb.size(0)
            loss_running += loss.item() * bs
            seen += bs

            train_loss_now = loss_running / max(seen, 1)
            train_nmse_db_now = nmse_to_db(train_loss_now)

            pbar.set_postfix(
                train_nmse=f"{train_loss_now:.3e}",
                train_db=f"{train_nmse_db_now:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        train_loss = loss_running / max(seen, 1)
        train_nmse_db = nmse_to_db(train_loss)

        # ====================================================
        # 每轮训练后，在总测试集上评估
        # best.pt 按 test_loss 最低保存
        # ====================================================
        total_test_loss, total_test_nmse_db = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            amp_enabled=cfg.amp,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        last_train_loss = train_loss
        last_train_nmse_db = train_nmse_db
        last_test_loss = total_test_loss
        last_test_nmse_db = total_test_nmse_db

        line = (
            f"epoch: {epoch}, "
            f"model_type: LLM4CP, "
            f"train_nmse: {train_loss:.6e} ({train_nmse_db:.2f} dB), "
            f"test_nmse: {total_test_loss:.6e} ({total_test_nmse_db:.2f} dB), "
            f"lr: {lr_now:.6g}"
        )

        print(line)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        # 每隔 save_every 轮保存一次 checkpoint
        if epoch % cfg.save_every == 0:
            save_checkpoint(
                path=os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                train_nmse_db=train_nmse_db,
                test_loss=total_test_loss,
                test_nmse_db=total_test_nmse_db,
                best_epoch=best_epoch,
                best_test_loss=best_test_loss,
                best_test_nmse_db=best_test_nmse_db,
            )

        # 按 test_loss 最低保存 best.pt
        if total_test_loss < best_test_loss:
            best_test_loss = total_test_loss
            best_test_nmse_db = total_test_nmse_db
            best_epoch = epoch

            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                train_loss=train_loss,
                train_nmse_db=train_nmse_db,
                test_loss=total_test_loss,
                test_nmse_db=total_test_nmse_db,
                best_epoch=best_epoch,
                best_test_loss=best_test_loss,
                best_test_nmse_db=best_test_nmse_db,
            )

    # ========================================================
    # 保存 last.pt
    # ========================================================
    last_path = os.path.join(run_dir, "last.pt")

    save_checkpoint(
        path=last_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=cfg.epochs,
        train_loss=last_train_loss,
        train_nmse_db=last_train_nmse_db,
        test_loss=last_test_loss,
        test_nmse_db=last_test_nmse_db,
        best_epoch=best_epoch,
        best_test_loss=best_test_loss,
        best_test_nmse_db=best_test_nmse_db,
    )

    print("\n训练完成。")
    print(f"输出目录：{run_dir}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best test NMSE loss: {best_test_loss:.6e}")
    print(f"Best test NMSE dB: {best_test_nmse_db:.2f} dB")
    print(f"Best model path: {best_path}")
    print(f"Last model path: {last_path}")


if __name__ == "__main__":
    main()