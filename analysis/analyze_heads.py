"""
CogVideoX-2b Attention Head Role Analysis

각 attention head가 temporal(시간축) vs spatial(공간축) 중 어느 쪽에 더 attend하는지 분석.
Sparse VideoGen 방법론을 따름.

측정 방식:
  - Temporal score: token (t0, h0, w0)이 같은 위치 (t, h0, w0)에 attend하는 평균 weight (t != t0)
  - Spatial score:  token (t0, h0, w0)이 같은 시간 (t0, h, w)에 attend하는 평균 weight ((h,w) != (h0,w0))
  - Temporal ratio = temporal / (temporal + spatial)

Usage:
    python analysis/analyze_heads.py
    python analysis/analyze_heads.py --num_frames 9 --num_inference_steps 10
"""

import argparse
import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Custom Attention Processor
# ─────────────────────────────────────────────────────────────────────────────

class HeadRoleAnalysisProcessor:
    """
    CogVideoXAttnProcessor2_0을 대체하여 per-head temporal/spatial 통계를 수집.
    메모리 절약을 위해 헤드별로 순차 처리.
    """

    def __init__(self, layer_idx: int, stats: dict, T: int, H: int, W: int):
        self.layer_idx = layer_idx
        self.stats     = stats
        self.T, self.H, self.W = T, H, W

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, H, W = self.T, self.H, self.W
        HW = H * W
        Nv = T * HW
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        batch_size, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key   = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim  = inner_dim // attn.heads
        num_heads = attn.heads
        scale     = head_dim ** -0.5

        query = query.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        key   = key.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, num_heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        # ── 헤드별 순차 처리 (메모리 절약) ────────────────────────────────────
        eye_HW = torch.eye(HW, device=hidden_states.device, dtype=query.dtype)
        eye_T  = torch.eye(T,  device=hidden_states.device, dtype=query.dtype)

        head_outputs     = []
        temporal_scores  = []
        spatial_scores   = []

        for h_idx in range(num_heads):
            q_h = query[:, h_idx]  # [B, seq, head_dim]
            k_h = key[:, h_idx]
            v_h = value[:, h_idx]

            # 전체 시퀀스 attention weight (text + video → text + video)
            attn_w = torch.softmax(
                (q_h @ k_h.transpose(-2, -1)) * scale, dim=-1
            )  # [B, seq, seq]

            # ── 통계 계산: video → video 부분만 사용 ─────────────────────────
            with torch.no_grad():
                # [Nv, Nv] (batch 평균)
                vid = attn_w[:, text_seq_length:, text_seq_length:].mean(0)
                # [T, HW, T, HW] = [t_src, s_src, t_tgt, s_tgt]
                a = vid.reshape(T, HW, T, HW)

                # Temporal: 같은 공간(s), 다른 시간(t)
                # temporal_mat[t0, t1] = sum_s a[t0, s, t1, s]
                temporal_mat = (a * eye_HW.unsqueeze(0).unsqueeze(2)).sum(dim=(1, 3))
                # 자기 자신(t0==t1) 제외
                t_score = (temporal_mat.sum() - temporal_mat.trace()).item() / Nv

                # Spatial: 같은 시간(t), 다른 공간(s)
                # spatial_mat[s0, s1] = sum_t a[t, s0, t, s1]
                spatial_mat = (a * eye_T.unsqueeze(1).unsqueeze(3)).sum(dim=(0, 2))
                # 자기 자신(s0==s1) 제외
                s_score = (spatial_mat.sum() - spatial_mat.trace()).item() / Nv

                temporal_scores.append(t_score)
                spatial_scores.append(s_score)

            # 헤드 출력
            head_outputs.append((attn_w @ v_h).unsqueeze(1))  # [B, 1, seq, head_dim]

        # ── 통계 누적 ─────────────────────────────────────────────────────────
        lk = self.layer_idx
        if lk not in self.stats:
            self.stats[lk] = {
                'temporal': [0.0] * num_heads,
                'spatial':  [0.0] * num_heads,
                'count': 0,
            }
        for h_idx in range(num_heads):
            self.stats[lk]['temporal'][h_idx] += temporal_scores[h_idx]
            self.stats[lk]['spatial'][h_idx]  += spatial_scores[h_idx]
        self.stats[lk]['count'] += 1

        # ── 출력 조합 ─────────────────────────────────────────────────────────
        hidden_states = torch.cat(head_outputs, dim=1)          # [B, heads, seq, head_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def build_ratio_matrix(stats: dict, num_layers: int, num_heads: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ratio    = np.full((num_layers, num_heads), 0.5)
    temporal = np.zeros((num_layers, num_heads))
    spatial  = np.zeros((num_layers, num_heads))

    for layer_idx in range(num_layers):
        if layer_idx not in stats:
            continue
        count = stats[layer_idx]['count']
        for h in range(num_heads):
            t = stats[layer_idx]['temporal'][h] / count
            s = stats[layer_idx]['spatial'][h]  / count
            total = t + s
            ratio[layer_idx, h]    = t / total if total > 0 else 0.5
            temporal[layer_idx, h] = t
            spatial[layer_idx, h]  = s

    return ratio, temporal, spatial


def plot_results(stats: dict, num_layers: int, num_heads: int, output_dir: str):
    ratio, temporal, spatial = build_ratio_matrix(stats, num_layers, num_heads)

    # ── 3-panel heatmap ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    panels = [
        (ratio,    'Temporal Ratio\n(1=temporal, 0=spatial)', 'RdBu_r', 0.0, 1.0),
        (temporal, 'Temporal Score (raw)',                    'Reds',   0.0, None),
        (spatial,  'Spatial Score (raw)',                     'Blues',  0.0, None),
    ]
    for ax, (data, title, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(data, cmap=cmap, aspect='auto',
                       vmin=vmin, vmax=vmax if vmax else data.max())
        ax.set_xlabel('Head index', fontsize=12)
        ax.set_ylabel('Layer index', fontsize=12)
        ax.set_title(title, fontsize=13)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(0, num_heads, 5))
        ax.set_yticks(range(0, num_layers, 5))

    plt.suptitle('CogVideoX-2b: Per-Head Temporal vs Spatial Attention Analysis', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'head_analysis_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[saved] {path}")
    plt.close()

    # ── Ratio distribution histogram ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ratio.flatten(), bins=25, edgecolor='black', color='steelblue', alpha=0.8)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1.5, label='neutral (0.5)')
    ax.set_xlabel('Temporal Ratio', fontsize=12)
    ax.set_ylabel('# Heads × Layers', fontsize=12)
    ax.set_title('Distribution of Temporal Ratios across All Heads', fontsize=13)
    ax.legend()
    path2 = os.path.join(output_dir, 'head_analysis_histogram.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    print(f"[saved] {path2}")
    plt.close()

    # ── Per-layer mean ratio line plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    mean_per_layer = ratio.mean(axis=1)
    std_per_layer  = ratio.std(axis=1)
    ax.plot(range(num_layers), mean_per_layer, marker='o', linewidth=2)
    ax.fill_between(range(num_layers),
                    mean_per_layer - std_per_layer,
                    mean_per_layer + std_per_layer, alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='neutral')
    ax.set_xlabel('Layer index', fontsize=12)
    ax.set_ylabel('Mean Temporal Ratio', fontsize=12)
    ax.set_title('Mean Temporal Ratio per Layer', fontsize=13)
    ax.set_xticks(range(num_layers))
    ax.legend()
    ax.grid(True, alpha=0.3)
    path3 = os.path.join(output_dir, 'head_analysis_per_layer.png')
    plt.savefig(path3, dpi=150, bbox_inches='tight')
    print(f"[saved] {path3}")
    plt.close()

    return ratio


def print_summary(ratio: np.ndarray, num_layers: int, num_heads: int):
    threshold = 0.65
    n_temporal = int((ratio > threshold).sum())
    n_spatial  = int((ratio < 1 - threshold).sum())
    n_mixed    = num_layers * num_heads - n_temporal - n_spatial

    print(f"\n{'='*55}")
    print(f"  Summary (threshold={threshold})")
    print(f"{'='*55}")
    print(f"  Temporal-dominant (ratio > {threshold}):  {n_temporal:3d} / {num_layers*num_heads}")
    print(f"  Spatial-dominant  (ratio < {1-threshold:.2f}): {n_spatial:3d} / {num_layers*num_heads}")
    print(f"  Mixed:                             {n_mixed:3d} / {num_layers*num_heads}")
    print(f"{'='*55}")

    print("\nTemporal-dominant heads per layer (ratio > 0.65):")
    for layer_idx in range(num_layers):
        heads = [h for h in range(num_heads) if ratio[layer_idx, h] > threshold]
        if heads:
            avg = ratio[layer_idx, heads].mean()
            print(f"  Layer {layer_idx:2d}: heads {heads}  (mean ratio={avg:.2f})")

    print("\nSpatial-dominant heads per layer (ratio < 0.35):")
    for layer_idx in range(num_layers):
        heads = [h for h in range(num_heads) if ratio[layer_idx, h] < 1 - threshold]
        if heads:
            avg = ratio[layer_idx, heads].mean()
            print(f"  Layer {layer_idx:2d}: heads {heads}  (mean ratio={avg:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CogVideoX-2b attention head role analysis')
    parser.add_argument('--model_path', type=str,
                        default='/home/sbjeon/.cache/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01')
    parser.add_argument('--prompt', type=str,
                        default='A dog playing guitar outdoors')
    parser.add_argument('--num_frames', type=int, default=9,
                        help='비디오 프레임 수 (메모리 절약을 위해 기본값 9)')
    parser.add_argument('--num_inference_steps', type=int, default=10,
                        help='분석에 사용할 denoising step 수 (많을수록 안정적)')
    parser.add_argument('--output_dir', type=str, default='analysis/results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype  = torch.bfloat16

    # ── 토큰 구조 계산 ─────────────────────────────────────────────────────────
    # temporal_compression_ratio=4, patch_size=2
    # latent spatial: sample_height=60, sample_width=90
    T  = (args.num_frames - 1) // 4 + 1
    H  = 60 // 2   # = 30
    W  = 90 // 2   # = 45
    Nv = T * H * W
    print(f"Video token grid : T={T}, H={H}, W={W} → {Nv} video tokens")
    print(f"Attention map    : {Nv}×{Nv} per head (video-only)")
    print(f"GPU memory est.  : {Nv*Nv*2/1024**2:.1f} MB per head in bfloat16\n")

    # ── 파이프라인 로드 ────────────────────────────────────────────────────────
    print("Loading CogVideoX-2b pipeline ...")
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
    ).to(device)
    pipe.transformer.eval()
    pipe.set_progress_bar_config(desc="Denoising", leave=True)

    # ── Custom processor 설치 ─────────────────────────────────────────────────
    stats = {}
    processors = {}
    for name, module in pipe.transformer.named_modules():
        parts = name.split('.')
        if 'transformer_blocks' in parts and name.endswith('attn1'):
            layer_idx = int(parts[parts.index('transformer_blocks') + 1])
            processors[f"{name}.processor"] = HeadRoleAnalysisProcessor(
                layer_idx, stats, T, H, W
            )

    pipe.transformer.set_attn_processor(processors)
    print(f"Installed analysis processors on {len(processors)} layers\n")

    # ── Denoising 실행 ─────────────────────────────────────────────────────────
    print(f"Running inference: '{args.prompt}'")
    print(f"num_frames={args.num_frames}, steps={args.num_inference_steps}\n")

    with torch.no_grad():
        pipe(
            prompt=args.prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(42),
            output_type="latent",   # VAE decode 생략
        )

    # ── 결과 저장 ──────────────────────────────────────────────────────────────
    stats_path = os.path.join(args.output_dir, 'head_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)
    print(f"\n[saved] {stats_path}")

    # ── 시각화 ────────────────────────────────────────────────────────────────
    num_layers = 30
    num_heads  = 30
    ratio = plot_results(stats, num_layers, num_heads, args.output_dir)
    print_summary(ratio, num_layers, num_heads)


if __name__ == '__main__':
    main()
