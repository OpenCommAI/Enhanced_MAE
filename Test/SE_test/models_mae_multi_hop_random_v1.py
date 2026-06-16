import math
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

    assert embed_dim % 2 == 0, "embed_dim must be even"

    gy = torch.arange(H, dtype=torch.float32, device=device)
    gx = torch.arange(W, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(gy, gx, indexing="ij")

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


class MAEChannelTargetBalancedHop(nn.Module):
    """
    支持 8toX 的增强 MAE 跳频模型。

    输入:
        x_input: (B, 2, 64, 32 * history_tti)
        y_label: (B, 2, 64, 32 * pred_tti)

    输出:
        pred:    (B, 2, 64, 32 * pred_tti)

    说明:
        history_tti 默认 8；
        pred_tti 可以是 1、2、3、...、8；
        未来区域使用 0 占位，不使用 y_label 作为输入，避免标签泄露。
    """

    def __init__(
        self,
        history_tti: int = 8,
        pred_tti: int = 1,
        in_chans: int = 2,
        embed_dim: int = 128,
        depth: int = 3,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 3,
        decoder_num_heads: int = 8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        hop_steps: Tuple[int, ...] = (1, 3, 5, 7),
        balanced_hop: bool = True,
    ):
        super().__init__()

        assert history_tti >= 1, "history_tti must be >= 1"
        assert pred_tti >= 1, "pred_tti must be >= 1"
        assert len(hop_steps) > 0, "hop_steps must not be empty"

        self.history_tti = history_tti
        self.pred_tti = pred_tti
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.hop_steps = tuple(int(h) for h in hop_steps)
        self.balanced_hop = bool(balanced_hop)

        H = 64
        W = 32 * (history_tti + pred_tti)

        self.img_size = (H, W)

        self.ph_c = 8
        self.pw_c = 32

        self.gh_c = H // self.ph_c
        self.gw_c = W // self.pw_c

        assert self.gh_c == 8
        assert self.gw_c == history_tti + pred_tti

        self.target_start_col = history_tti

        for h in self.hop_steps:
            assert 1 <= h < self.gh_c, f"invalid hop_step={h}"
            assert math.gcd(self.gh_c, h) == 1, (
                f"hop_step={h} must be coprime with Gh={self.gh_c}"
            )

        self.register_buffer(
            "hop_candidates",
            torch.tensor(self.hop_steps, dtype=torch.long),
            persistent=False,
        )

        # coarse patch: 每个 TTI 对应 8 个 coarse patch
        self.patch_c = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(self.ph_c, self.pw_c),
            stride=(self.ph_c, self.pw_c),
            bias=True,
        )

        # fine-W patch
        self.ph_w = 8
        self.pw_w = 16
        self.gh_w = H // self.ph_w
        self.gw_w = W // self.pw_w

        assert self.gh_w == 8
        assert self.gw_w == 2 * (history_tti + pred_tti)

        self.patch_w = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(self.ph_w, self.pw_w),
            stride=(self.ph_w, self.pw_w),
            bias=True,
        )

        # fine-H patch
        self.ph_h = 4
        self.pw_h = 32
        self.gh_h = H // self.ph_h
        self.gw_h = W // self.pw_h

        assert self.gh_h == 16
        assert self.gw_h == history_tti + pred_tti

        self.patch_h = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=(self.ph_h, self.pw_h),
            stride=(self.ph_h, self.pw_h),
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

        self.register_buffer(
            "pos_w",
            build_2d_sincos_pos_embed(
                embed_dim,
                (self.gh_w, self.gw_w),
                cls_token=False,
                device="cpu",
            ),
            persistent=False,
        )

        self.register_buffer(
            "pos_h",
            build_2d_sincos_pos_embed(
                embed_dim,
                (self.gh_h, self.gw_h),
                cls_token=False,
                device="cpu",
            ),
            persistent=False,
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim, eps=1e-6),
                nn.MultiheadAttention(embed_dim, num_heads, batch_first=True),
                nn.LayerNorm(embed_dim, eps=1e-6),
                nn.Sequential(
                    nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                ),
            )
            for _ in range(depth)
        ])

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

        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(decoder_embed_dim, eps=1e-6),
                nn.MultiheadAttention(decoder_embed_dim, decoder_num_heads, batch_first=True),
                nn.LayerNorm(decoder_embed_dim, eps=1e-6),
                nn.Sequential(
                    nn.Linear(decoder_embed_dim, int(decoder_embed_dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(decoder_embed_dim * mlp_ratio), decoder_embed_dim),
                ),
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.out_patch_dim = in_chans * self.ph_c * self.pw_c
        self.head = nn.Linear(decoder_embed_dim, self.out_patch_dim)

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

    def _sample_balanced_hops(
        self,
        B: int,
        device,
        generator: Optional[torch.Generator] = None,
        force_hop_step: Optional[int] = None,
    ):
        candidates = self.hop_candidates.to(device)
        num_hops = candidates.numel()

        if force_hop_step is not None:
            h = int(force_hop_step)
            assert h in self.hop_steps, f"force_hop_step={h} is not in hop_steps={self.hop_steps}"
            return torch.full((B, 1), h, device=device, dtype=torch.long)

        if not self.balanced_hop:
            idx = torch.randint(
                0,
                num_hops,
                (B, 1),
                device=device,
                generator=generator,
            )
            return candidates[idx]

        if B >= num_hops:
            base_count = B // num_hops
            remainder = B % num_hops

            hops = candidates.repeat_interleave(base_count)

            if remainder > 0:
                perm_extra = torch.randperm(
                    num_hops,
                    device=device,
                    generator=generator,
                )
                hops = torch.cat([hops, candidates[perm_extra[:remainder]]], dim=0)

            perm = torch.randperm(
                B,
                device=device,
                generator=generator,
            )
            hops = hops[perm]

            return hops.view(B, 1)

        perm = torch.randperm(
            num_hops,
            device=device,
            generator=generator,
        )

        return candidates[perm[:B]].view(B, 1)

    def _sample_rows_coarse(
        self,
        B: int,
        device,
        generator: Optional[torch.Generator] = None,
        force_hop_step: Optional[int] = None,
    ):
        Gh = self.gh_c

        cols = torch.arange(
            0,
            self.history_tti,
            device=device,
            dtype=torch.long,
        )

        start = torch.randint(
            0,
            Gh,
            (B, 1),
            device=device,
            generator=generator,
        )

        hop_step = self._sample_balanced_hops(
            B=B,
            device=device,
            generator=generator,
            force_hop_step=force_hop_step,
        )

        rows = (start + hop_step * cols.view(1, -1)) % Gh

        return rows, cols, start, hop_step

    def _coarse_query_and_mask(
        self,
        x_concat: torch.Tensor,
        rows_c: torch.Tensor,
        cols_c: torch.Tensor,
    ):
        B = x_concat.size(0)
        device = x_concat.device

        x = self.patch_c(x_concat).flatten(2).transpose(1, 2)
        x = x + self.pos_c[:, 1:, :].to(device)

        keep_idx = rows_c * self.gw_c + cols_c.view(1, -1)

        L = self.gh_c * self.gw_c

        mask_full = torch.ones(B, L, dtype=torch.bool, device=device)
        mask_full.scatter_(1, keep_idx, False)

        keep_sorted, _ = torch.sort(keep_idx, dim=1)

        all_idx = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

        keep_map = torch.zeros(B, L, dtype=torch.bool, device=device)
        keep_map.scatter_(1, keep_sorted, True)

        mask_idx_sorted = all_idx[~keep_map].view(B, -1)

        ids_cat = torch.cat([keep_sorted, mask_idx_sorted], dim=1)
        ids_restore = torch.argsort(ids_cat, dim=1)

        x_keep = gather_tokens(x, keep_sorted)

        cls = (self.cls_token + self.pos_c[:, :1, :]).to(device).expand(B, -1, -1)
        q_tokens = torch.cat([cls, x_keep], dim=1)

        return q_tokens, mask_full, ids_restore

    def _fineW_tokens(
        self,
        x_concat: torch.Tensor,
        rows_c: torch.Tensor,
        cols_c: torch.Tensor,
    ):
        device = x_concat.device

        x = self.patch_w(x_concat).flatten(2).transpose(1, 2)
        x = x + self.pos_w.to(device)

        cols_pair = torch.stack(
            [2 * cols_c, 2 * cols_c + 1],
            dim=1,
        ).reshape(-1).to(device)

        rows_rep = rows_c.repeat_interleave(2, dim=1)

        keep_idx = rows_rep * self.gw_w + cols_pair.view(1, -1)
        keep_sorted, _ = torch.sort(keep_idx, dim=1)

        return gather_tokens(x, keep_sorted)

    def _fineH_tokens(
        self,
        x_concat: torch.Tensor,
        rows_c: torch.Tensor,
        cols_c: torch.Tensor,
    ):
        device = x_concat.device

        x = self.patch_h(x_concat).flatten(2).transpose(1, 2)
        x = x + self.pos_h.to(device)

        rows_pair = torch.stack(
            [2 * rows_c, 2 * rows_c + 1],
            dim=2,
        ).reshape(rows_c.size(0), -1)

        cols_rep = cols_c.view(1, -1).repeat_interleave(2, dim=1)

        keep_idx = rows_pair * self.gw_h + cols_rep
        keep_sorted, _ = torch.sort(keep_idx, dim=1)

        return gather_tokens(x, keep_sorted)

    def forward_encoder_multi(
        self,
        x_concat: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        force_hop_step: Optional[int] = None,
    ):
        B = x_concat.size(0)
        device = x_concat.device

        rows_c, cols_c, start_rows, hop_steps = self._sample_rows_coarse(
            B,
            device,
            generator=generator,
            force_hop_step=force_hop_step,
        )

        q_tokens, mask_coarse, ids_restore = self._coarse_query_and_mask(
            x_concat,
            rows_c,
            cols_c,
        )

        t_w = self._fineW_tokens(x_concat, rows_c, cols_c)
        t_h = self._fineH_tokens(x_concat, rows_c, cols_c)

        x = torch.cat([q_tokens, t_w, t_h], dim=1)
        x = self.pre_norm(x)

        for seq in self.blocks:
            ln1, mha, ln2, mlp = seq

            x_res = x
            x = ln1(x)
            x, _ = mha(x, x, x, need_weights=False)
            x = x_res + x

            x_res = x
            x = ln2(x)
            x = x_res + mlp(x)

        x = self.post_norm(x)

        latent = x[:, :1 + self.history_tti, :]

        return latent, mask_coarse, ids_restore, rows_c, start_rows, hop_steps

    def forward_decoder(
        self,
        latent: torch.Tensor,
        ids_restore: torch.Tensor,
    ):
        x = self.decoder_embed(latent)

        B, Lk, Cdec = x.shape

        L = self.gh_c * self.gw_c
        mask_len = L - (Lk - 1)

        mask_tokens = self.mask_token.expand(B, mask_len, Cdec)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)

        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, Cdec),
        )

        x = torch.cat([x[:, :1, :], x_], dim=1)

        x = x + self.pos_dec.to(x.device)

        for seq in self.decoder_blocks:
            ln1, mha, ln2, mlp = seq

            x_res = x
            x = ln1(x)
            x, _ = mha(x, x, x, need_weights=False)
            x = x_res + x

            x_res = x
            x = ln2(x)
            x = x_res + mlp(x)

        x = self.decoder_norm(x)

        x = x[:, 1:, :].view(B, self.gh_c, self.gw_c, Cdec)

        return x

    def decode_target_columns(self, x_grid: torch.Tensor):
        """
        x_grid:
            (B, gh_c, gw_c, decoder_dim)

        输出:
            (B, 2, 64, 32 * pred_tti)
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
        y_label: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        force_hop_step: Optional[int] = None,
    ):
        """
        x_input:
            (B, 2, 64, 32 * history_tti)

        y_label:
            (B, 2, 64, 32 * pred_tti)
        """
        B = x_input.size(0)
        device = x_input.device

        assert x_input.dim() == 4, f"x_input dim error: {x_input.shape}"
        assert x_input.shape[1] == self.in_chans
        assert x_input.shape[2] == 64
        assert x_input.shape[3] == 32 * self.history_tti, (
            f"x_input width error: got {x_input.shape[3]}, "
            f"expected {32 * self.history_tti}"
        )

        if y_label is not None:
            assert y_label.dim() == 4, f"y_label dim error: {y_label.shape}"
            assert y_label.shape[1] == self.in_chans
            assert y_label.shape[2] == 64
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

        latent, mask, ids_restore, rows_c, start_rows, hop_steps = self.forward_encoder_multi(
            x_concat,
            generator=generator,
            force_hop_step=force_hop_step,
        )

        x_grid = self.forward_decoder(latent, ids_restore)
        pred = self.decode_target_columns(x_grid)

        if y_label is None:
            return pred

        nmse_mean, nmse_db = nmse_and_db(pred, y_label)

        loss_dict: Dict[str, torch.Tensor] = {
            "nmse": nmse_mean,
            "nmse_db": nmse_db,
            "hop_steps": hop_steps.detach().view(-1),
            "start_rows": start_rows.detach().view(-1),
        }

        return nmse_mean, pred, mask, loss_dict


MAEChannelTargetHop3 = MAEChannelTargetBalancedHop


def build_mae_channel_target_hop3(
    history_tti: int = 8,
    pred_tti: int = 1,
    **kwargs,
) -> MAEChannelTargetBalancedHop:
    hop_steps = kwargs.pop("hop_steps", (1, 3, 5, 7))
    balanced_hop = kwargs.pop("balanced_hop", True)

    return MAEChannelTargetBalancedHop(
        history_tti=history_tti,
        pred_tti=pred_tti,
        in_chans=2,
        embed_dim=128,
        depth=3,
        num_heads=8,
        mlp_ratio=4.0,
        decoder_embed_dim=256,
        decoder_depth=3,
        decoder_num_heads=8,
        hop_steps=hop_steps,
        balanced_hop=balanced_hop,
        **kwargs,
    )