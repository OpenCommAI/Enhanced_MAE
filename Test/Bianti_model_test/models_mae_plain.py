import torch
import torch.nn as nn
from functools import partial
from typing import Optional, Tuple, Dict


def build_2d_sincos_pos_embed(
    embed_dim: int,
    grid_hw: Tuple[int, int],
    cls_token: bool = False,
    device=None,
):
    H, W = grid_hw
    device = device or torch.device("cpu")

    grid_y = torch.arange(H, dtype=torch.float32, device=device)
    grid_x = torch.arange(W, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(grid_y, grid_x, indexing="ij")

    def pe1d(dim, pos):
        assert dim % 2 == 0
        omega = torch.arange(dim // 2, dtype=torch.float32, device=device)
        omega = 1.0 / (10000 ** (omega / (dim / 2)))
        out = torch.einsum("p,d->pd", pos.reshape(-1), omega)
        return torch.cat([torch.sin(out), torch.cos(out)], dim=1)

    emb_y = pe1d(embed_dim // 2, gy)
    emb_x = pe1d(embed_dim // 2, gx)
    emb = torch.cat([emb_y, emb_x], dim=1)

    if cls_token:
        cls = torch.zeros(1, embed_dim, device=device)
        emb = torch.cat([cls, emb], dim=0)

    return emb.unsqueeze(0)


def nmse_and_db(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12):
    diff = (pred - target).reshape(pred.size(0), -1)
    tgt = target.reshape(target.size(0), -1)

    nmse = diff.pow(2).sum(dim=1) / (tgt.pow(2).sum(dim=1) + eps)
    nmse_mean = nmse.mean()
    nmse_db = 10.0 * torch.log10(torch.clamp(nmse_mean, min=eps))

    return nmse_mean, nmse_db


def gather_tokens(x_tokens: torch.Tensor, keep_idx_sorted: torch.Tensor):
    return torch.gather(
        x_tokens,
        dim=1,
        index=keep_idx_sorted.unsqueeze(-1).expand(-1, -1, x_tokens.size(-1)),
    )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h

        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h

        return x


class MAEChannelPlainAblation(nn.Module):
    """
    普通 MAE 消融模型：
    - 没有跳频采样；
    - 没有 fine-W / fine-H 多尺度融合；
    - 只使用 coarse patch；
    - 历史区域随机保留 keep_patches 个 patch；
    - 未来目标区域全部 mask；
    - 未来区域使用 0 占位，避免标签泄露。

    输入:
        x_input: (B, 2, 64, 32 * history_tti)
        y_label: (B, 2, 64, 32 * pred_tti)

    输出:
        pred: (B, 2, 64, 32 * pred_tti)

    默认:
        history_tti = 8
        pred_tti = 1
        keep_patches = 8

    当 history_tti=8 时：
        历史区域 patch 数 = 8 * 8 = 64
        keep_patches=8 等价于 mask_ratio=0.875
    """

    def __init__(
        self,
        history_tti: int = 8,
        pred_tti: int = 1,
        keep_patches: int = 8,
        in_chans: int = 2,
        embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 3,
        decoder_num_heads: int = 8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        assert history_tti >= 1, "history_tti must be >= 1"
        assert pred_tti >= 1, "pred_tti must be >= 1"
        assert keep_patches >= 1, "keep_patches must be >= 1"

        self.history_tti = history_tti
        self.pred_tti = pred_tti
        self.keep_patches = keep_patches
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        H = 64
        W = 32 * (history_tti + pred_tti)

        self.img_size = (H, W)
        self.target_start_col = history_tti

        self.ph_c = 8
        self.pw_c = 32

        self.gh_c = H // self.ph_c
        self.gw_c = W // self.pw_c

        assert self.gh_c == 8
        assert self.gw_c == history_tti + pred_tti

        self.total_patch_num = self.gh_c * self.gw_c
        self.history_patch_num = self.gh_c * self.history_tti

        assert keep_patches <= self.history_patch_num, (
            f"keep_patches={keep_patches} 不能大于历史 patch 数 {self.history_patch_num}"
        )

        self.mask_ratio = 1.0 - float(keep_patches) / float(self.history_patch_num)

        self.patch_c = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(self.ph_c, self.pw_c),
            stride=(self.ph_c, self.pw_c),
            bias=True,
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.register_buffer(
            "pos_c",
            build_2d_sincos_pos_embed(
                embed_dim,
                (self.gh_c, self.gw_c),
                cls_token=True,
                device="cpu",
            ),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(depth)
            ]
        )

        self.pre_norm = norm_layer(embed_dim)
        self.post_norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.register_buffer(
            "pos_dec",
            build_2d_sincos_pos_embed(
                decoder_embed_dim,
                (self.gh_c, self.gw_c),
                cls_token=True,
                device="cpu",
            ),
            persistent=False,
        )

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=decoder_embed_dim,
                    num_heads=decoder_num_heads,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.out_patch_dim = in_chans * self.ph_c * self.pw_c
        self.head = nn.Linear(decoder_embed_dim, self.out_patch_dim)

        # 历史区域索引。
        # Conv2d flatten 后 token 顺序为 row-major:
        # index = row * grid_w + col
        rows = torch.arange(self.gh_c).view(-1, 1)
        cols = torch.arange(self.history_tti).view(1, -1)
        history_indices = rows * self.gw_c + cols
        history_indices = history_indices.reshape(-1).long()

        self.register_buffer(
            "history_indices",
            history_indices,
            persistent=False,
        )

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

            if getattr(m, "bias", None) is not None:
                nn.init.constant_(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)

    def random_mask_history(
        self,
        x_tokens: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        """
        普通 MAE 随机掩码方式。

        只在历史区域中随机保留 keep_patches 个 token。
        未来区域不参与可见 token，始终为 mask。

        输入:
            x_tokens: (B, L, C)

        输出:
            x_keep:      (B, keep_patches, C)
            mask_full:   (B, L), False 表示可见，True 表示 mask
            ids_restore: (B, L)
            keep_sorted: (B, keep_patches)
        """
        B, L, C = x_tokens.shape
        device = x_tokens.device

        assert L == self.total_patch_num

        history_indices = self.history_indices.to(device)
        history_patch_num = history_indices.numel()

        noise = torch.rand(
            B,
            history_patch_num,
            device=device,
            generator=generator,
        )

        keep_pos = noise.topk(
            k=self.keep_patches,
            dim=1,
            largest=True,
            sorted=False,
        ).indices

        keep_idx = history_indices[keep_pos]
        keep_sorted, _ = torch.sort(keep_idx, dim=1)

        mask_full = torch.ones(B, L, dtype=torch.bool, device=device)
        mask_full.scatter_(1, keep_sorted, False)

        all_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        keep_map = torch.zeros(B, L, dtype=torch.bool, device=device)
        keep_map.scatter_(1, keep_sorted, True)

        mask_idx_sorted = all_idx[~keep_map].view(B, -1)

        ids_cat = torch.cat([keep_sorted, mask_idx_sorted], dim=1)
        ids_restore = torch.argsort(ids_cat, dim=1)

        x_keep = gather_tokens(x_tokens, keep_sorted)

        return x_keep, mask_full, ids_restore, keep_sorted

    def forward_encoder(
        self,
        x_concat: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ):
        B = x_concat.size(0)
        device = x_concat.device

        x = self.patch_c(x_concat).flatten(2).transpose(1, 2)
        x = x + self.pos_c[:, 1:, :].to(device)

        x_keep, mask, ids_restore, keep_indices = self.random_mask_history(
            x,
            generator=generator,
        )

        cls = self.cls_token + self.pos_c[:, :1, :].to(device)
        cls = cls.expand(B, -1, -1)

        x = torch.cat([cls, x_keep], dim=1)

        x = self.pre_norm(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.post_norm(x)

        return x, mask, ids_restore, keep_indices

    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ):
        x = self.decoder_embed(latent)

        B, Lk, Cdec = x.shape

        L = self.total_patch_num
        keep_len_without_cls = Lk - 1
        mask_len = L - keep_len_without_cls

        mask_tokens = self.mask_token.expand(B, mask_len, Cdec)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)

        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, Cdec),
        )

        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.pos_dec.to(x.device)

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)

        x = x[:, 1:, :].view(B, self.gh_c, self.gw_c, Cdec)

        return x

    def decode_target_columns(self, x_grid: torch.Tensor):
        """
        从 decoder 输出的完整 coarse 网格中，只取未来目标列。

        输出:
            pred: (B, 2, 64, 32 * pred_tti)
        """
        B = x_grid.size(0)

        tgt = x_grid[
            :,
            :,
            self.target_start_col:self.target_start_col + self.pred_tti,
            :,
        ]

        tgt = self.head(tgt)

        C = self.in_chans

        tgt = tgt.view(
            B,
            self.gh_c,
            self.pred_tti,
            C,
            self.ph_c,
            self.pw_c,
        )

        tgt = tgt.permute(0, 3, 1, 4, 2, 5).contiguous()

        tgt = tgt.view(
            B,
            C,
            self.gh_c * self.ph_c,
            self.pred_tti * self.pw_c,
        )

        return tgt

    def forward(
        self,
        x_input: torch.Tensor,
        y_label: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        force_hop_step: Optional[int] = None,
    ):
        """
        force_hop_step 只是为了兼容其他训练代码接口。
        普通 MAE 不使用跳频，因此该参数不会生效。
        """
        B = x_input.size(0)
        device = x_input.device

        assert x_input.dim() == 4
        assert y_label.dim() == 4

        assert x_input.shape[1] == self.in_chans
        assert y_label.shape[1] == self.in_chans

        assert x_input.shape[2] == 64
        assert y_label.shape[2] == 64

        assert x_input.shape[3] == 32 * self.history_tti, (
            f"x_input width error: got {x_input.shape[3]}, "
            f"expected {32 * self.history_tti}"
        )

        assert y_label.shape[3] == 32 * self.pred_tti, (
            f"y_label width error: got {y_label.shape[3]}, "
            f"expected {32 * self.pred_tti}"
        )

        future_placeholder = torch.zeros(
            B,
            self.in_chans,
            64,
            32 * self.pred_tti,
            dtype=x_input.dtype,
            device=device,
        )

        x_concat = torch.cat([x_input, future_placeholder], dim=3)

        latent, mask, ids_restore, keep_indices = self.forward_encoder(
            x_concat,
            generator=generator,
        )

        x_grid = self.forward_decoder(latent, ids_restore)
        pred = self.decode_target_columns(x_grid)

        nmse_mean, nmse_db = nmse_and_db(pred, y_label)

        loss_dict: Dict[str, torch.Tensor] = {
            "nmse": nmse_mean,
            "nmse_db": nmse_db,
            "keep_indices": keep_indices.detach(),
        }

        return nmse_mean, pred, mask, loss_dict


def build_mae_channel_plain_mae(
    history_tti: int = 8,
    pred_tti: int = 1,
    keep_patches: int = 8,
    **kwargs,
) -> MAEChannelPlainAblation:
    return MAEChannelPlainAblation(
        history_tti=history_tti,
        pred_tti=pred_tti,
        keep_patches=keep_patches,
        in_chans=2,
        embed_dim=128,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        decoder_embed_dim=256,
        decoder_depth=3,
        decoder_num_heads=8,
        **kwargs,
    )