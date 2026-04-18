"""
Head-Wise LoRA for CogVideoX-2b Subject Motion Customization

analyze_heads.py 분석 결과를 기반으로 두 어댑터를 head 단위로 분리:

  id_adapter     → spatial head의 Q, K만 업데이트  (identity 보존)
  motion_adapter → temporal head의 V, Out만 업데이트 (motion 학습)

이로써 temporal head의 Q/K (attention 패턴)를 id_adapter가 오염하지 않고,
spatial head의 V/Out을 motion_adapter가 건드리지 않는 완전한 역할 분리 달성.

분석 결과 (ratio > 0.7):
  TEMPORAL_HEADS: {4:[3,11], 5:[4], 7:[24,25], 10:[17], 11:[1,4,25], ...}
  SPATIAL_HEADS : 각 레이어별 temporal이 아닌 나머지 헤드

파라미터 비교 (r=32 기준):
  id_adapter     Standard: 7,372,800  →  Head-Wise: 7,286,784  (절감  1.2%, but Q/K 오염 차단)
  motion_adapter Standard: 7,372,800  →  Head-Wise: 1,323,008  (절감 91.0%)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from safetensors.torch import save_file, load_file


# ─────────────────────────────────────────────────────────────────────────────
# 분석 결과: Head 목록
# ─────────────────────────────────────────────────────────────────────────────

# Temporal heads (ratio > 0.7): motion 담당
TEMPORAL_HEADS: Dict[int, List[int]] = {
    4:  [3, 11],
    5:  [4],
    7:  [24, 25],
    10: [17],
    11: [1, 4, 25],
    12: [15, 21],
    21: [5],
    22: [13, 17, 22, 28],
    23: [12, 14],
    25: [20],
    27: [19, 24],
}

# Spatial heads: 각 레이어에서 temporal이 아닌 나머지 → identity 담당
def _build_spatial_heads(
    num_layers: int = 30,
    num_heads: int = 30,
    temporal: Dict[int, List[int]] = TEMPORAL_HEADS,
) -> Dict[int, List[int]]:
    result = {}
    for l in range(num_layers):
        t_set = set(temporal.get(l, []))
        result[l] = [h for h in range(num_heads) if h not in t_set]
    return result

SPATIAL_HEADS: Dict[int, List[int]] = _build_spatial_heads()

HEAD_DIM   = 64
NUM_HEADS  = 30
HIDDEN_DIM = NUM_HEADS * HEAD_DIM   # 1920


# ─────────────────────────────────────────────────────────────────────────────
# Head-Wise LoRA Linear
# ─────────────────────────────────────────────────────────────────────────────

class HeadWiseLoRALinear(nn.Module):
    """
    특정 head slice에만 LoRA delta를 적용하는 Linear 래퍼.

    mode='row' (to_q, to_k, to_v 용):
        output의 [h*head_dim : (h+1)*head_dim] 행만 업데이트.
        lora_A: [r, in_features]
        lora_B: [n_heads*head_dim, r]
        delta  = lora_B @ lora_A  →  [n_heads*head_dim, in_features]
        out[..., head_indices] += x @ lora_A.T @ lora_B.T * scaling

    mode='col' (to_out 용):
        input의 [h*head_dim : (h+1)*head_dim] 열만 읽어서 delta 계산.
        lora_A: [r, n_heads*head_dim]
        lora_B: [out_features, r]
        out += x[..., head_indices] @ lora_A.T @ lora_B.T * scaling
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        target_heads: List[int],
        head_dim: int,
        rank: int,
        lora_alpha: float,
        mode: str = 'row',
    ):
        super().__init__()
        self.in_features   = base_linear.in_features
        self.out_features  = base_linear.out_features
        self.target_heads  = sorted(target_heads)
        self.head_dim      = head_dim
        self.rank          = rank
        self.scaling       = lora_alpha / rank
        self.mode          = mode

        n_h = len(target_heads)

        # 기본 weight/bias 참조 (frozen, 메모리 공유)
        self.weight = base_linear.weight
        self.bias   = base_linear.bias

        # LoRA 파라미터 (학습 대상)
        if mode == 'row':
            self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
            self.lora_B = nn.Parameter(torch.zeros(n_h * head_dim, rank))
        elif mode == 'col':
            self.lora_A = nn.Parameter(torch.empty(rank, n_h * head_dim))
            self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        else:
            raise ValueError(f"mode must be 'row' or 'col', got '{mode}'")

        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

        # head index → flat tensor  (register_buffer: device 이동 자동)
        indices = []
        for h in self.target_heads:
            indices.extend(range(h * head_dim, (h + 1) * head_dim))
        self.register_buffer('head_indices', torch.tensor(indices, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)

        if self.mode == 'row':
            delta = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out[..., self.head_indices] = out[..., self.head_indices] + delta
        elif self.mode == 'col':
            x_h   = x[..., self.head_indices]
            delta = F.linear(F.linear(x_h, self.lora_A), self.lora_B) * self.scaling
            out   = out + delta

        return out

    def extra_repr(self) -> str:
        return (f"mode={self.mode}, heads={self.target_heads}, "
                f"rank={self.rank}, scaling={self.scaling:.3f}, "
                f"in={self.in_features}, out={self.out_features}")


# ─────────────────────────────────────────────────────────────────────────────
# 주입 함수
# ─────────────────────────────────────────────────────────────────────────────

def _get_block(transformer: nn.Module, layer_idx: int) -> Optional[nn.Module]:
    for name, module in transformer.named_modules():
        if name == f'transformer_blocks.{layer_idx}':
            return module
    return None


def apply_id_adapter_head_wise(
    transformer: nn.Module,
    spatial_heads: Dict[int, List[int]] = SPATIAL_HEADS,
    rank: int = 32,
    lora_alpha: float = 32.0,
    head_dim: int = HEAD_DIM,
) -> Dict[str, HeadWiseLoRALinear]:
    """
    id_adapter: spatial head의 to_q, to_k에만 LoRA 적용.
    temporal head의 Q/K attention 패턴을 보호.
    """
    injected = {}

    for layer_idx, heads in spatial_heads.items():
        block = _get_block(transformer, layer_idx)
        if block is None or not hasattr(block, 'attn1'):
            continue

        for proj_name in ('to_q', 'to_k'):
            orig = getattr(block.attn1, proj_name)
            hw   = HeadWiseLoRALinear(orig, heads, head_dim, rank, lora_alpha, mode='row')
            setattr(block.attn1, proj_name, hw)
            key = f'transformer_blocks.{layer_idx}.attn1.{proj_name}'
            injected[key] = hw

    _print_injection_summary('id_adapter (spatial heads → Q, K)', injected, transformer)
    return injected


def apply_motion_adapter_head_wise(
    transformer: nn.Module,
    temporal_heads: Dict[int, List[int]] = TEMPORAL_HEADS,
    rank: int = 32,
    lora_alpha: float = 32.0,
    head_dim: int = HEAD_DIM,
) -> Dict[str, HeadWiseLoRALinear]:
    """
    motion_adapter: temporal head의 to_v, to_out에만 LoRA 적용.
    """
    injected = {}

    for layer_idx, heads in temporal_heads.items():
        block = _get_block(transformer, layer_idx)
        if block is None or not hasattr(block, 'attn1'):
            continue

        # to_v (mode='row')
        orig = block.attn1.to_v
        hw   = HeadWiseLoRALinear(orig, heads, head_dim, rank, lora_alpha, mode='row')
        block.attn1.to_v = hw
        injected[f'transformer_blocks.{layer_idx}.attn1.to_v'] = hw

        # to_out[0] (mode='col')
        orig = block.attn1.to_out[0]
        hw   = HeadWiseLoRALinear(orig, heads, head_dim, rank, lora_alpha, mode='col')
        block.attn1.to_out[0] = hw
        injected[f'transformer_blocks.{layer_idx}.attn1.to_out.0'] = hw

    _print_injection_summary('motion_adapter (temporal heads → V, Out)', injected, transformer)
    return injected


def apply_both_adapters(
    transformer: nn.Module,
    id_rank: int = 32,
    id_alpha: float = 32.0,
    motion_rank: int = 32,
    motion_alpha: float = 32.0,
    head_dim: int = HEAD_DIM,
) -> Tuple[Dict[str, HeadWiseLoRALinear], Dict[str, HeadWiseLoRALinear]]:
    """
    id_adapter + motion_adapter 동시 주입.

    주입 순서:
      1. motion_adapter (temporal V, Out) 먼저 주입
      2. id_adapter (spatial Q, K) 주입
      → 두 어댑터가 서로 다른 projection을 담당하므로 충돌 없음

    Returns:
        id_injected, motion_injected
    """
    print("=" * 55)
    print("  Head-Wise LoRA 전체 주입")
    print("=" * 55)

    motion_injected = apply_motion_adapter_head_wise(
        transformer, TEMPORAL_HEADS, motion_rank, motion_alpha, head_dim
    )
    id_injected = apply_id_adapter_head_wise(
        transformer, SPATIAL_HEADS, id_rank, id_alpha, head_dim
    )

    # 학습 대상 파라미터 설정 (lora_A, lora_B만)
    for param in transformer.parameters():
        param.requires_grad = False
    for module in list(id_injected.values()) + list(motion_injected.values()):
        module.lora_A.requires_grad = True
        module.lora_B.requires_grad = True

    n_id     = sum(p.numel() for m in id_injected.values()
                   for p in [m.lora_A, m.lora_B])
    n_motion = sum(p.numel() for m in motion_injected.values()
                   for p in [m.lora_A, m.lora_B])
    print(f"\n  id_adapter     학습 파라미터: {n_id:>12,}")
    print(f"  motion_adapter 학습 파라미터: {n_motion:>12,}")
    print(f"  합계:                         {n_id+n_motion:>12,}")
    print("=" * 55)

    return id_injected, motion_injected


def _print_injection_summary(name, injected, transformer):
    n_params = sum(p.numel() for m in injected.values()
                   for p in [m.lora_A, m.lora_B])
    print(f"[HeadWiseLoRA] {name}")
    print(f"  → {len(injected)}개 모듈 주입, LoRA 파라미터: {n_params:,}")


# ─────────────────────────────────────────────────────────────────────────────
# 저장 / 불러오기
# ─────────────────────────────────────────────────────────────────────────────

def save_head_wise_lora(
    injected: Dict[str, HeadWiseLoRALinear],
    save_path: str,
    adapter_name: str,
    rank: int,
    lora_alpha: float,
    head_map: Dict[int, List[int]],
    mode: str,
):
    """LoRA 가중치를 safetensors로 저장."""
    os.makedirs(save_path, exist_ok=True)

    state_dict = {}
    for path, module in injected.items():
        state_dict[f"{path}.lora_A.weight"] = module.lora_A.data.cpu()
        state_dict[f"{path}.lora_B.weight"] = module.lora_B.data.cpu()

    weights_path = os.path.join(save_path, "pytorch_lora_weights.safetensors")
    save_file(state_dict, weights_path)

    config = {
        "adapter_name": adapter_name,
        "lora_type":    "head_wise",
        "mode":         mode,
        "rank":         rank,
        "lora_alpha":   lora_alpha,
        "head_dim":     HEAD_DIM,
        "head_map":     {str(k): v for k, v in head_map.items()},
    }
    with open(os.path.join(save_path, "head_wise_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"  [saved] {weights_path}  ({len(state_dict)} tensors)")


def save_both_adapters(
    id_injected:     Dict[str, HeadWiseLoRALinear],
    motion_injected: Dict[str, HeadWiseLoRALinear],
    output_dir: str,
    id_rank: int = 32,
    id_alpha: float = 32.0,
    motion_rank: int = 32,
    motion_alpha: float = 32.0,
):
    save_head_wise_lora(
        id_injected,
        save_path=os.path.join(output_dir, "id_adapter"),
        adapter_name="id_adapter",
        rank=id_rank,
        lora_alpha=id_alpha,
        head_map=SPATIAL_HEADS,
        mode="row",
    )
    save_head_wise_lora(
        motion_injected,
        save_path=os.path.join(output_dir, "motion_adapter"),
        adapter_name="motion_adapter",
        rank=motion_rank,
        lora_alpha=motion_alpha,
        head_map=TEMPORAL_HEADS,
        mode="row_col",
    )


def load_head_wise_lora(
    transformer: nn.Module,
    load_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, HeadWiseLoRALinear]:
    """저장된 Head-Wise LoRA를 Transformer에 로드."""
    with open(os.path.join(load_path, "head_wise_config.json")) as f:
        config = json.load(f)

    rank       = config["rank"]
    lora_alpha = config["lora_alpha"]
    head_dim   = config["head_dim"]
    head_map   = {int(k): v for k, v in config["head_map"].items()}
    adapter    = config["adapter_name"]

    if adapter == "id_adapter":
        injected = apply_id_adapter_head_wise(transformer, head_map, rank, lora_alpha, head_dim)
    else:
        injected = apply_motion_adapter_head_wise(transformer, head_map, rank, lora_alpha, head_dim)

    state_dict = load_file(
        os.path.join(load_path, "pytorch_lora_weights.safetensors"), device=device
    )
    for path, module in injected.items():
        if (key := f"{path}.lora_A.weight") in state_dict:
            module.lora_A.data = state_dict[key].to(dtype)
        if (key := f"{path}.lora_B.weight") in state_dict:
            module.lora_B.data = state_dict[key].to(dtype)

    print(f"[loaded] {adapter} Head-Wise LoRA from {load_path}")
    return injected


# ─────────────────────────────────────────────────────────────────────────────
# 파라미터 비교 유틸
# ─────────────────────────────────────────────────────────────────────────────

def print_parameter_comparison(
    id_rank:     int = 32,
    motion_rank: int = 32,
    std_rank:    int = 32,
):
    hd, nd, hidden = HEAD_DIM, NUM_HEADS, HIDDEN_DIM

    # Standard
    std_id     = 30 * 2 * 2 * (std_rank * hidden)   # 30 layers, Q+K, A+B
    std_motion = 30 * 2 * 2 * (std_rank * hidden)   # 30 layers, V+Out, A+B

    # Head-Wise id (spatial Q, K, mode=row, all 30 layers)
    hw_id = sum(
        2 * (id_rank * hidden + len(SPATIAL_HEADS[l]) * hd * id_rank)
        for l in range(30)
    )
    # Head-Wise motion (temporal V+Out)
    hw_motion = sum(
        (motion_rank * hidden + len(h) * hd * motion_rank) +   # to_v  (row)
        (motion_rank * len(h) * hd + hidden * motion_rank)     # to_out (col)
        for h in TEMPORAL_HEADS.values()
    )

    print("=" * 62)
    print(f"  {'':25s} {'Standard':>12}  {'Head-Wise':>12}  {'절감':>6}")
    print("=" * 62)
    print(f"  {'id_adapter (Q,K)':25s} {std_id:>12,}  {hw_id:>12,}  "
          f"{(1-hw_id/std_id)*100:>5.1f}%")
    print(f"  {'motion_adapter (V,Out)':25s} {std_motion:>12,}  {hw_motion:>12,}  "
          f"{(1-hw_motion/std_motion)*100:>5.1f}%")
    print("-" * 62)
    print(f"  {'합계':25s} {std_id+std_motion:>12,}  {hw_id+hw_motion:>12,}  "
          f"{(1-(hw_id+hw_motion)/(std_id+std_motion))*100:>5.1f}%")
    print("=" * 62)
    print("  ※ id_adapter 절감은 미미하지만")
    print("     temporal head의 Q/K 오염을 차단하는 효과가 핵심")
