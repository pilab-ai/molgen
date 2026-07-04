import math
from typing import Dict, Optional, Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    时间嵌入：把 t ∈ [0, 1] 编码成高维向量
    类似扩散模型中的 sinusoidal time embedding。

    输入:
        t: [B]，每个样本对应一个时间，范围通常是 [0, 1]

    输出:
        emb: [B, dim]
    """

    def __init__(self, dim: int):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even.")

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: [B]
        """
        if t.ndim == 1:
            t = t[:, None]  # [B, 1]

        half_dim = self.dim // 2
        device = t.device

        # 频率项
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device, dtype=t.dtype)
            / max(half_dim - 1, 1)
        )  # [half_dim]

        args = t * freq[None, :]  # [B, half_dim]

        emb = torch.cat(
            [torch.sin(args), torch.cos(args)],
            dim=-1
        )  # [B, dim]

        return emb

class MoleculeBFNBackbone(nn.Module):
    """
    BFN 的神经网络部分。

    它接收当前分布参数:
        1. coord_mu: 当前坐标分布均值 [B, N, 3]
        2. coord_log_rho: 当前坐标分布 precision 的 log 值 [B, N, 1]
        3. atom_probs: 当前原子类型概率 [B, N, K]
        4. t: 时间 [B]
        5. mask: 有效原子 mask [B, N]

    输出:
        1. coord_pred: 预测的干净坐标 [B, N, 3]
        2. atom_logits: 预测的原子类型 logits [B, N, K]
    """

    def __init__(
        self,
        num_atom_types: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_atom_types = num_atom_types

        self.time_emb = SinusoidalTimeEmbedding(time_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_dim),
        )

        # 输入特征:
        # coord_mu: 3
        # coord_log_rho: 1
        # atom_probs: K
        # time embedding: time_dim
        in_dim = 3 + 1 + num_atom_types + time_dim

        self.in_proj = nn.Linear(in_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        self.coord_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

        self.atom_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_atom_types),
        )

    def forward(
        self,
        coord_mu: torch.Tensor,
        coord_log_rho: torch.Tensor,
        atom_probs: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        coord_mu: [B, N, 3]
        coord_log_rho: [B, N, 1]
        atom_probs: [B, N, K]
        t: [B]
        mask: [B, N], True 表示真实原子，False 表示 padding
        """

        B, N, _ = coord_mu.shape

        # 时间嵌入
        t_emb = self.time_emb(t)          # [B, time_dim]
        t_emb = self.time_mlp(t_emb)      # [B, time_dim]
        t_emb = t_emb[:, None, :].expand(B, N, -1)  # [B, N, time_dim]

        # 拼接当前 BFN 分布参数
        h = torch.cat(
            [
                coord_mu,
                coord_log_rho,
                atom_probs,
                t_emb,
            ],
            dim=-1,
        )  # [B, N, 3 + 1 + K + time_dim]

        h = self.in_proj(h)  # [B, N, hidden_dim]

        if mask is not None:
            # Transformer 中 True 表示需要被忽略的位置
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        h = self.encoder(
            h,
            src_key_padding_mask=key_padding_mask,
        )  # [B, N, hidden_dim]

        if mask is not None:
            h = h * mask[:, :, None].float()

        coord_pred = self.coord_head(h)  # [B, N, 3]
        atom_logits = self.atom_head(h)  # [B, N, K]

        return coord_pred, atom_logits


# ============================================================
# 3. BFN 主模型
# ============================================================

class MoleculeBFN(nn.Module):
    """
    同时处理连续坐标和离散原子类型的 BFN。

    连续坐标:
        使用高斯 input distribution:
            x_i ~ N(mu_i, 1 / rho_i)

    离散原子类型:
        使用 categorical input distribution:
            h_i ~ Cat(p_i)

    注意:
        这是一个教学版 / 实用版 BFN。
        损失使用 MSE + CE，便于理解和训练。
    """

    def __init__(
        self,
        num_atom_types: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 8,
        beta1_coord: float = 25.0,
        alpha1_atom: float = 10.0,
        coord_loss_weight: float = 1.0,
        atom_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.num_atom_types = num_atom_types

        # 坐标最终累计 precision
        # 越大表示 t=1 时输入分布越接近真实坐标
        self.beta1_coord = beta1_coord

        # 原子类型最终累计 evidence strength
        self.alpha1_atom = alpha1_atom

        self.coord_loss_weight = coord_loss_weight
        self.atom_loss_weight = atom_loss_weight

        self.backbone = MoleculeBFNBackbone(
            num_atom_types=num_atom_types,
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )

    # --------------------------------------------------------
    # 3.1 坐标 accuracy / precision schedule
    # --------------------------------------------------------

    def beta_coord(self, t: torch.Tensor) -> torch.Tensor:
        """
        坐标变量的累计 accuracy。

        t = 0 时 beta = 0，表示几乎没有关于真实坐标的信息。
        t = 1 时 beta = beta1_coord，表示信息较充分。

        这里使用 t^2 schedule。
        """
        return self.beta1_coord * t ** 2

    # --------------------------------------------------------
    # 3.2 离散原子类型 evidence schedule
    # --------------------------------------------------------

    def alpha_atom(self, t: torch.Tensor) -> torch.Tensor:
        """
        离散原子类型的累计 evidence strength。

        t = 0 时 alpha = 0，原子类型接近均匀分布。
        t = 1 时 alpha = alpha1_atom，原子类型分布更确定。
        """
        return self.alpha1_atom * t ** 2

    # --------------------------------------------------------
    # 3.3 构造连续坐标的 BFN input distribution
    # --------------------------------------------------------

    def make_continuous_input(
        self,
        coords: torch.Tensor,
        t: torch.Tensor,
    ):
        """
        根据真实坐标 coords 和时间 t 构造当前 input distribution 参数。

        假设 prior:
            x ~ N(0, I)

        sender observation:
            y ~ N(x, 1 / beta)

        贝叶斯更新后:
            rho_t = 1 + beta
            mu_t = (beta * x + sqrt(beta) * eps) / (1 + beta)

        输入:
            coords: [B, N, 3]
            t: [B]

        输出:
            coord_mu: [B, N, 3]
            coord_log_rho: [B, N, 1]
        """

        B, N, _ = coords.shape

        beta = self.beta_coord(t)  # [B]
        beta = beta.view(B, 1, 1)  # [B, 1, 1]

        noise = torch.randn_like(coords)

        # 避免显式构造 y = x + eps / sqrt(beta)
        # 因为 beta 很小时 y 数值会非常大
        coord_mu = (
            beta * coords + torch.sqrt(beta.clamp_min(1e-12)) * noise
        ) / (1.0 + beta)

        coord_rho = 1.0 + beta  # [B, 1, 1]

        coord_log_rho = torch.log(coord_rho).expand(B, N, 1)

        return coord_mu, coord_log_rho

    # --------------------------------------------------------
    # 3.4 构造离散原子类型的 BFN input distribution
    # --------------------------------------------------------

    def make_discrete_input(
        self,
        atom_types: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        根据真实原子类型和时间 t 构造当前 categorical input distribution。

        atom_types:
            [B, N]，每个位置是 0 ~ K-1 的类别编号

        输出:
            atom_probs: [B, N, K]

        简化理解:
            t 小时，atom_probs 接近均匀分布；
            t 大时，atom_probs 更接近真实 one-hot。
        """

        B, N = atom_types.shape
        K = self.num_atom_types

        alpha = self.alpha_atom(t)      # [B]
        alpha = alpha.view(B, 1, 1)     # [B, 1, 1]

        one_hot = F.one_hot(
            atom_types.clamp(min=0),
            num_classes=K,
        ).float()  # [B, N, K]

        noise = torch.randn_like(one_hot)

        # 离散 BFN 的一个常用构造：
        # 正确类别方向获得更大的 evidence，其余类别获得较小 evidence
        #
        # 当 alpha = 0:
        #     logits 约为 0，softmax 后接近均匀分布
        #
        # 当 alpha 变大:
        #     正确类别 logit 更大，softmax 后更接近 one-hot
        evidence_logits = (
            alpha * (K * one_hot - 1.0)
            + torch.sqrt((alpha * K).clamp_min(1e-12)) * noise
        )  # [B, N, K]

        atom_probs = F.softmax(evidence_logits, dim=-1)

        # padding 位置设为均匀分布
        if mask is not None:
            uniform = torch.full_like(atom_probs, 1.0 / K)
            atom_probs = torch.where(
                mask[:, :, None].bool(),
                atom_probs,
                uniform,
            )

        return atom_probs

    # --------------------------------------------------------
    # 3.5 训练损失
    # --------------------------------------------------------

    def loss(
        self,
        coords: torch.Tensor,
        atom_types: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        训练一步的 loss。

        输入:
            coords: [B, N, 3]，建议提前中心化并缩放到较稳定范围，如 [-1, 1]
            atom_types: [B, N]，类别编号，范围 0 ~ K-1
            mask: [B, N]，True 表示真实原子，False 表示 padding

        输出:
            一个 dict，包含总 loss 和各项 loss。
        """

        device = coords.device
        B, N, _ = coords.shape

        # 每个 batch 随机采样一个时间 t
        # 避免 t=0 或 t=1 的边界数值问题
        t = torch.rand(B, device=device)
        t = t.clamp(1e-4, 1.0 - 1e-4)

        # 构造当前 BFN input distribution
        coord_mu, coord_log_rho = self.make_continuous_input(coords, t)
        atom_probs = self.make_discrete_input(atom_types, t, mask)

        # 神经网络预测 clean 坐标和原子类型
        coord_pred, atom_logits = self.backbone(
            coord_mu=coord_mu,
            coord_log_rho=coord_log_rho,
            atom_probs=atom_probs,
            t=t,
            mask=mask,
        )

        # ----------------------------
        # 坐标 MSE，只在真实原子上计算
        # ----------------------------
        mask_float = mask[:, :, None].float()  # [B, N, 1]

        coord_se = (coord_pred - coords) ** 2  # [B, N, 3]
        coord_se = coord_se * mask_float

        loss_coord = coord_se.sum() / (
            mask_float.sum().clamp_min(1.0) * 3.0
        )

        # ----------------------------
        # 原子类型 CE，只在真实原子上计算
        # ----------------------------
        # atom_logits: [B, N, K]
        # cross_entropy 需要 [B, K, N]
        ce = F.cross_entropy(
            atom_logits.transpose(1, 2),
            atom_types,
            reduction="none",
        )  # [B, N]

        loss_atom = (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)

        loss = (
            self.coord_loss_weight * loss_coord
            + self.atom_loss_weight * loss_atom
        )

        return {
            "loss": loss,
            "loss_coord": loss_coord.detach(),
            "loss_atom": loss_atom.detach(),
        }

    # --------------------------------------------------------
    # 3.6 采样生成
    # --------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_nodes: Union[int, Sequence[int], torch.Tensor],
        num_steps: int = 100,
        stochastic_atom: bool = True,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        从 prior 开始逐步生成分子。

        输入:
            batch_size:
                生成多少个分子

            num_nodes:
                如果是 int，表示每个分子都有相同原子数；
                如果是 list / tensor，表示每个分子的原子数可以不同。

            num_steps:
                BFN 更新步数。越大通常越稳定，但越慢。

            stochastic_atom:
                True: 原子类型按概率采样；
                False: 原子类型取 argmax。

        输出:
            coords: [B, N, 3]
            atom_types: [B, N]
            atom_probs: [B, N, K]
            mask: [B, N]
        """

        if device is None:
            device = next(self.parameters()).device

        K = self.num_atom_types

        # ----------------------------------------
        # 构造 mask
        # ----------------------------------------
        if isinstance(num_nodes, int):
            N = num_nodes
            mask = torch.ones(batch_size, N, dtype=torch.bool, device=device)
        else:
            counts = torch.as_tensor(num_nodes, dtype=torch.long, device=device)
            batch_size = counts.numel()
            N = int(counts.max().item())

            arange = torch.arange(N, device=device)[None, :]
            mask = arange < counts[:, None]  # [B, N]

        B = batch_size

        # ----------------------------------------
        # 初始化 continuous input distribution
        # prior: x ~ N(0, I)
        # mu = 0, rho = 1
        # ----------------------------------------
        coord_mu = torch.zeros(B, N, 3, device=device)
        coord_rho = torch.ones(B, N, 1, device=device)

        # ----------------------------------------
        # 初始化 discrete input distribution
        # prior: uniform categorical
        # 使用 evidence logits 累积证据
        # ----------------------------------------
        atom_evidence_logits = torch.zeros(B, N, K, device=device)
        atom_probs = torch.full(
            (B, N, K),
            1.0 / K,
            device=device,
        )

        for step in range(num_steps):
            t0 = step / num_steps
            t1 = (step + 1) / num_steps
            tm = (step + 0.5) / num_steps

            t_mid = torch.full((B,), tm, device=device)

            coord_log_rho = torch.log(coord_rho.clamp_min(1e-12))

            # 当前网络预测
            coord_pred, atom_logits = self.backbone(
                coord_mu=coord_mu,
                coord_log_rho=coord_log_rho,
                atom_probs=atom_probs,
                t=t_mid,
                mask=mask,
            )

            # =====================================================
            # A. 连续坐标的 Bayesian update
            # =====================================================

            beta0 = self.beta_coord(
                torch.tensor(t0, device=device)
            )
            beta1 = self.beta_coord(
                torch.tensor(t1, device=device)
            )

            d_beta = beta1 - beta0
            d_beta = d_beta.clamp_min(1e-8)

            # 根据模型预测的 clean coord 构造一个 sender sample
            # y ~ N(coord_pred, 1 / d_beta)
            y_coord = coord_pred + torch.randn_like(coord_pred) / torch.sqrt(d_beta)

            # 高斯-高斯贝叶斯更新:
            # rho_new = rho_old + d_beta
            # mu_new = (rho_old * mu_old + d_beta * y) / rho_new
            coord_rho_new = coord_rho + d_beta

            coord_mu = (
                coord_rho * coord_mu + d_beta * y_coord
            ) / coord_rho_new

            coord_rho = coord_rho_new

            # padding 位置清零
            coord_mu = coord_mu * mask[:, :, None].float()

            # =====================================================
            # B. 离散原子类型的 Bayesian-style evidence update
            # =====================================================

            atom_pred_probs = F.softmax(atom_logits, dim=-1)  # [B, N, K]

            if stochastic_atom:
                # 按预测概率采样类别
                flat_probs = atom_pred_probs.reshape(B * N, K)
                atom_sample = torch.multinomial(
                    flat_probs,
                    num_samples=1,
                ).view(B, N)
            else:
                # 直接取最大概率类别
                atom_sample = atom_pred_probs.argmax(dim=-1)

            atom_one_hot = F.one_hot(
                atom_sample,
                num_classes=K,
            ).float()  # [B, N, K]

            alpha0 = self.alpha_atom(
                torch.tensor(t0, device=device)
            )
            alpha1 = self.alpha_atom(
                torch.tensor(t1, device=device)
            )

            d_alpha = alpha1 - alpha0
            d_alpha = d_alpha.clamp_min(1e-8)

            # 当前步新增 evidence
            atom_step_evidence = (
                d_alpha * (K * atom_one_hot - 1.0)
                + torch.sqrt(d_alpha * K) * torch.randn_like(atom_one_hot)
            )  # [B, N, K]

            atom_evidence_logits = atom_evidence_logits + atom_step_evidence

            atom_probs = F.softmax(atom_evidence_logits, dim=-1)

            # padding 位置设为均匀分布
            uniform = torch.full_like(atom_probs, 1.0 / K)
            atom_probs = torch.where(
                mask[:, :, None],
                atom_probs,
                uniform,
            )

        # 最后再用 t=1 的网络输出作为最终结果
        t_final = torch.ones(B, device=device)

        coord_log_rho = torch.log(coord_rho.clamp_min(1e-12))

        coord_final, atom_logits_final = self.backbone(
            coord_mu=coord_mu,
            coord_log_rho=coord_log_rho,
            atom_probs=atom_probs,
            t=t_final,
            mask=mask,
        )

        atom_probs_final = F.softmax(atom_logits_final, dim=-1)

        if stochastic_atom:
            flat_probs = atom_probs_final.reshape(B * N, K)
            atom_types = torch.multinomial(
                flat_probs,
                num_samples=1,
            ).view(B, N)
        else:
            atom_types = atom_probs_final.argmax(dim=-1)

        coord_final = coord_final * mask[:, :, None].float()

        return {
            "coords": coord_final,
            "atom_types": atom_types,
            "atom_probs": atom_probs_final,
            "mask": mask,
        }