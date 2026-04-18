"""
CogVideoX-2b Attention Head Role Analysis - Timestep별 분석

denoising timestep에 따라 각 head의 temporal/spatial 역할이 어떻게 변하는지 분석.

diffusion model에서 timestep은 노이즈 수준을 의미:
  - 높은 timestep (초반): 고노이즈 → 글로벌 구조/레이아웃 결정
  - 낮은 timestep (후반): 저노이즈 → 세부 디테일/텍스처 결정

Usage:
    python analysis/analyze_timesteps.py
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Timestep-aware Attention Processor
# ─────────────────────────────────────────────────────────────────────────────

class TimestepAwareProcessor:
    """
    denoising step index를 외부에서 주입받아 timestep별 통계를 분리 저장.
    stats 구조: stats[step_idx][layer_idx] = {temporal, spatial, timestep_value}
    """

    def __init__(self, layer_idx: int, stats: dict, step_ref: list, T: int, H: int, W: int):
        self.layer_idx = layer_idx
        self.stats     = stats
        self.step_ref  = step_ref   # [현재 step index] — 외부에서 갱신
        self.T, self.H, self.W = T, H, W

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T, H, W   = self.T, self.H, self.W
        HW        = H * W
        Nv        = T * HW
        step_idx  = self.step_ref[0]
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

        # ── Per-head 통계 수집 ────────────────────────────────────────────────
        eye_HW = torch.eye(HW, device=hidden_states.device, dtype=query.dtype)
        eye_T  = torch.eye(T,  device=hidden_states.device, dtype=query.dtype)

        head_outputs    = []
        temporal_scores = []
        spatial_scores  = []

        for h_idx in range(num_heads):
            q_h = query[:, h_idx]
            k_h = key[:, h_idx]
            v_h = value[:, h_idx]

            attn_w = torch.softmax(
                (q_h @ k_h.transpose(-2, -1)) * scale, dim=-1
            )

            with torch.no_grad():
                vid = attn_w[:, text_seq_length:, text_seq_length:].mean(0)
                a   = vid.reshape(T, HW, T, HW)

                temporal_mat = (a * eye_HW.unsqueeze(0).unsqueeze(2)).sum(dim=(1, 3))
                t_score = (temporal_mat.sum() - temporal_mat.trace()).item() / Nv

                spatial_mat = (a * eye_T.unsqueeze(1).unsqueeze(3)).sum(dim=(0, 2))
                s_score = (spatial_mat.sum() - spatial_mat.trace()).item() / Nv

                temporal_scores.append(t_score)
                spatial_scores.append(s_score)

            head_outputs.append((attn_w @ v_h).unsqueeze(1))

        # ── stats[step_idx][layer_idx] 에 저장 ───────────────────────────────
        if step_idx not in self.stats:
            self.stats[step_idx] = {}
        lk = self.layer_idx
        if lk not in self.stats[step_idx]:
            self.stats[step_idx][lk] = {
                'temporal': [0.0] * num_heads,
                'spatial':  [0.0] * num_heads,
            }
        for h in range(num_heads):
            self.stats[step_idx][lk]['temporal'][h] += temporal_scores[h]
            self.stats[step_idx][lk]['spatial'][h]  += spatial_scores[h]

        # ── 출력 조합 ─────────────────────────────────────────────────────────
        hidden_states = torch.cat(head_outputs, dim=1)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, num_heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


# ─────────────────────────────────────────────────────────────────────────────
# Ratio 계산
# ─────────────────────────────────────────────────────────────────────────────

def compute_ratio_per_step(stats, num_steps, num_layers, num_heads):
    """
    ratio[step, layer, head] 반환
    """
    ratio = np.full((num_steps, num_layers, num_heads), 0.5)
    for step in range(num_steps):
        if step not in stats:
            continue
        for layer in range(num_layers):
            if layer not in stats[step]:
                continue
            for h in range(num_heads):
                t = stats[step][layer]['temporal'][h]
                s = stats[step][layer]['spatial'][h]
                total = t + s
                ratio[step, layer, h] = t / total if total > 0 else 0.5
    return ratio


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

def plot_timestep_analysis(ratio, timesteps_values, num_layers, num_heads, output_dir):
    num_steps = len(timesteps_values)
    step_labels = [f"t={int(t)}" for t in timesteps_values]

    # ── 1. Step별 전체 평균 ratio 변화 (레이어 평균) ──────────────────────────
    mean_per_step = ratio.mean(axis=(1, 2))   # [num_steps]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(num_steps), mean_per_step, marker='o', linewidth=2, color='steelblue')
    ax.set_xticks(range(num_steps))
    ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Denoising Step (high noise → low noise)', fontsize=12)
    ax.set_ylabel('Mean Temporal Ratio (all layers & heads)', fontsize=12)
    ax.set_title('전체 평균 Temporal Ratio vs Timestep', fontsize=13)
    ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='neutral')
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = os.path.join(output_dir, 'ts_overall_mean.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] {path}")

    # ── 2. Layer별 temporal ratio의 timestep 변화 히트맵 ─────────────────────
    # ratio를 layer 기준으로 head 평균: [num_steps, num_layers]
    ratio_by_layer = ratio.mean(axis=2)   # [num_steps, num_layers]

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(ratio_by_layer.T, aspect='auto', cmap='RdBu_r', vmin=0, vmax=0.5)
    ax.set_xticks(range(num_steps))
    ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([f'L{l}' for l in range(num_layers)], fontsize=8)
    ax.set_xlabel('Denoising Step', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Layer별 Temporal Ratio 변화 (head 평균)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Temporal Ratio')
    plt.tight_layout()
    path = os.path.join(output_dir, 'ts_layer_heatmap.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] {path}")

    # ── 3. Temporal-dominant head들의 timestep별 ratio 추적 ───────────────────
    # 이전 분석에서 확인된 top temporal heads
    key_heads = [
        (11, 1,  'L11-H1'),
        (4,  11, 'L4-H11'),
        (23, 14, 'L23-H14'),
        (22, 28, 'L22-H28'),
        (21, 5,  'L21-H5'),
        (10, 17, 'L10-H17'),
        (7,  25, 'L7-H25'),
        (7,  24, 'L7-H24'),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_heads)))
    for (layer, head, label), color in zip(key_heads, colors):
        ratios_over_time = ratio[:, layer, head]
        ax.plot(range(num_steps), ratios_over_time, marker='o', linewidth=2,
                label=label, color=color)
    ax.axhline(0.65, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='threshold (0.65)')
    ax.axhline(0.5,  color='red',  linestyle='--', linewidth=1, alpha=0.5, label='neutral')
    ax.set_xticks(range(num_steps))
    ax.set_xticklabels(step_labels, rotation=45, ha='right', fontsize=9)
    ax.set_xlabel('Denoising Step', fontsize=12)
    ax.set_ylabel('Temporal Ratio', fontsize=12)
    ax.set_title('Top Temporal Heads의 Timestep별 역할 변화', fontsize=13)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'ts_key_heads_tracking.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] {path}")

    # ── 4. 초반/중반/후반 timestep 비교 heatmap ──────────────────────────────
    thirds = [
        (0,             num_steps // 3,     "초반 (고노이즈)"),
        (num_steps // 3, 2 * num_steps // 3, "중반"),
        (2 * num_steps // 3, num_steps,      "후반 (저노이즈)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    for ax, (s, e, title) in zip(axes, thirds):
        data = ratio[s:e].mean(axis=0)   # [num_layers, num_heads]
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', vmin=0, vmax=0.6)
        ax.set_title(f'{title}\n(steps {s}~{e-1})', fontsize=12)
        ax.set_xlabel('Head', fontsize=10)
        ax.set_ylabel('Layer', fontsize=10)
        ax.set_xticks(range(0, num_heads, 5))
        ax.set_yticks(range(0, num_layers, 5))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle('Denoising 구간별 Temporal Ratio 분포 비교', fontsize=14)
    plt.tight_layout()
    path = os.path.join(output_dir, 'ts_phase_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] {path}")

    return ratio_by_layer


def print_timestep_summary(ratio, timesteps_values, num_layers, num_heads):
    num_steps = len(timesteps_values)
    print(f"\n{'='*60}")
    print("  Timestep별 Temporal-dominant Head 수 (ratio > 0.65)")
    print(f"{'='*60}")
    for step in range(num_steps):
        n = int((ratio[step] > 0.65).sum())
        t_val = int(timesteps_values[step])
        bar = '█' * n
        print(f"  Step {step:2d} (t={t_val:4d}): {n:3d}개  {bar}")

    # 어느 step에서 temporal ratio가 가장 높은가?
    mean_per_step = ratio.mean(axis=(1, 2))
    peak_step = int(np.argmax(mean_per_step))
    print(f"\n  → 전체 평균 temporal ratio가 가장 높은 step: "
          f"Step {peak_step} (t={int(timesteps_values[peak_step])})")

    # Layer별로 가장 temporal이 강한 timestep
    print(f"\n{'='*60}")
    print("  Layer별 가장 temporal이 강한 timestep")
    print(f"{'='*60}")
    for l in range(num_layers):
        layer_mean_per_step = ratio[:, l, :].mean(axis=1)  # [num_steps]
        peak = int(np.argmax(layer_mean_per_step))
        peak_val = layer_mean_per_step[peak]
        if peak_val > 0.05:
            print(f"  Layer {l:2d}: Step {peak:2d} (t={int(timesteps_values[peak]):4d})  "
                  f"ratio={peak_val:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline callback for step tracking
# ─────────────────────────────────────────────────────────────────────────────

def make_step_callback(step_ref, timestep_log):
    def callback(pipe, step_index, timestep, callback_kwargs):
        step_ref[0] = step_index
        timestep_log.append(float(timestep))
        return callback_kwargs
    return callback


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='/home/sbjeon/.cache/huggingface/hub/models--THUDM--CogVideoX-2b/snapshots/1137dacfc2c9c012bed6a0793f4ecf2ca8e7ba01')
    parser.add_argument('--prompt', type=str,
                        default='A dog playing guitar outdoors')
    parser.add_argument('--num_frames', type=int, default=9)
    parser.add_argument('--num_inference_steps', type=int, default=20,
                        help='timestep 해상도를 높이려면 늘릴수록 좋음 (권장 20)')
    parser.add_argument('--output_dir', type=str, default='analysis/results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    dtype  = torch.bfloat16

    T  = (args.num_frames - 1) // 4 + 1
    H  = 60 // 2
    W  = 90 // 2
    print(f"Video token grid: T={T}, H={H}, W={W} → {T*H*W} video tokens")

    # ── 파이프라인 로드 ────────────────────────────────────────────────────────
    print("Loading CogVideoX-2b pipeline ...")
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_path, torch_dtype=dtype
    ).to(device)
    pipe.transformer.eval()

    # ── Step 추적 설정 ─────────────────────────────────────────────────────────
    step_ref     = [0]      # 현재 step index (processor들이 공유)
    timestep_log = []       # 각 step의 실제 timestep 값

    # ── Custom processor 설치 ─────────────────────────────────────────────────
    stats      = {}
    processors = {}
    for name, module in pipe.transformer.named_modules():
        parts = name.split('.')
        if 'transformer_blocks' in parts and name.endswith('attn1'):
            layer_idx = int(parts[parts.index('transformer_blocks') + 1])
            processors[f"{name}.processor"] = TimestepAwareProcessor(
                layer_idx, stats, step_ref, T, H, W
            )
    pipe.transformer.set_attn_processor(processors)
    print(f"Installed {len(processors)} timestep-aware processors\n")

    # ── Denoising 실행 ─────────────────────────────────────────────────────────
    print(f"Running {args.num_inference_steps} steps ...")
    with torch.no_grad():
        pipe(
            prompt=args.prompt,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device=device).manual_seed(42),
            output_type="latent",
            callback_on_step_end=make_step_callback(step_ref, timestep_log),
            callback_on_step_end_tensor_inputs=['latents'],
        )

    num_steps  = len(timestep_log)
    num_layers = 30
    num_heads  = 30
    print(f"\nCollected {num_steps} timesteps: {[int(t) for t in timestep_log]}")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    stats_path = os.path.join(args.output_dir, 'timestep_stats.json')
    with open(stats_path, 'w') as f:
        json.dump({str(k): {str(lk): lv for lk, lv in v.items()}
                   for k, v in stats.items()}, f, indent=2)
    print(f"[saved] {stats_path}")

    ts_path = os.path.join(args.output_dir, 'timestep_values.json')
    with open(ts_path, 'w') as f:
        json.dump(timestep_log, f)
    print(f"[saved] {ts_path}")

    # ── 분석 및 시각화 ─────────────────────────────────────────────────────────
    ratio = compute_ratio_per_step(stats, num_steps, num_layers, num_heads)
    plot_timestep_analysis(ratio, timestep_log, num_layers, num_heads, args.output_dir)
    print_timestep_summary(ratio, timestep_log, num_layers, num_heads)


if __name__ == '__main__':
    main()
