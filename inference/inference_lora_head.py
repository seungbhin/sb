"""
inference_lora_head.py

Head-Wise LoRA (analysis/head_wise_lora.py) 기반 inference 스크립트.

기존 lora_inference.py와의 차이:
  - PEFT pipe.load_lora_weights() 대신 head_wise_lora.load_head_wise_lora() 사용
  - id_adapter   : spatial head Q/K 만 업데이트된 LoRA 적용
  - motion_adapter: temporal head V/Out 만 업데이트된 LoRA 적용
  - 두 어댑터 폴더에 head_wise_config.json이 있으면 Head-Wise 방식으로,
    없으면 기존 PEFT 방식으로 자동 fallback

Usage:
    cd /home/sbjeon/workspace2/sb
    python inference/inference_lora_head.py \
        --lora_dir  train/output/train_lora/dog_guitar \
        --prompt    "A sks dog is playing guitar on stage under spotlights" \
        --id_scale 1.0 --motion_scale 1.0
"""

import os
import sys
import json
import torch
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from analysis.head_wise_lora import (
    load_head_wise_lora,
    HeadWiseLoRALinear,
)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def load_train_args(lora_dir: str) -> dict:
    path = os.path.join(lora_dir, "train_args.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def is_head_wise(adapter_path: str) -> bool:
    """해당 경로가 Head-Wise LoRA인지 (head_wise_config.json 존재 여부로 판단)."""
    return os.path.exists(os.path.join(adapter_path, "head_wise_config.json"))


def set_head_wise_scale(injected: dict, scale: float):
    """
    HeadWiseLoRALinear 모듈들의 scaling 값을 조정.
    scaling = (alpha / rank) * scale
    """
    for module in injected.values():
        if isinstance(module, HeadWiseLoRALinear):
            base_scaling = module.lora_alpha_over_rank if hasattr(module, 'lora_alpha_over_rank') \
                           else module.scaling   # 기존 scaling = alpha/rank
            module.scaling = base_scaling * scale


def set_scaling_from_config(injected: dict, config_path: str, scale: float):
    """config에서 alpha/rank 읽어서 scale 반영."""
    with open(config_path) as f:
        cfg = json.load(f)
    base = cfg["lora_alpha"] / cfg["rank"]
    for module in injected.values():
        if isinstance(module, HeadWiseLoRALinear):
            module.scaling = base * scale


def print_lora_info(id_injected: dict, motion_injected: dict,
                    id_scale: float, motion_scale: float):
    n_id     = sum(p.numel() for m in id_injected.values()
                   for p in [m.lora_A, m.lora_B])
    n_motion = sum(p.numel() for m in motion_injected.values()
                   for p in [m.lora_A, m.lora_B])
    print("\n=== Head-Wise LoRA 정보 ===")
    print(f"  id_adapter     : {len(id_injected):3d} 모듈  "
          f"params={n_id:,}  scale={id_scale}")
    print(f"  motion_adapter : {len(motion_injected):3d} 모듈  "
          f"params={n_motion:,}  scale={motion_scale}")


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="CogVideoX Head-Wise LoRA Inference"
    )

    # 필수
    parser.add_argument("--lora_dir", type=str, required=True,
                        help="학습 출력 디렉토리 (id_adapter/, motion_adapter/ 포함)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="생성할 영상 프롬프트")

    # 모델 경로
    parser.add_argument("--model_path", type=str, default=None,
                        help="미지정 시 train_args.json에서 자동 로드")

    # 결과 저장
    parser.add_argument("--results_dir", type=str, default="inference/results",
                        help="결과 저장 루트 (기본값: inference/results)")

    # LoRA 스케일
    parser.add_argument("--id_scale",     type=float, default=1.0,
                        help="id_adapter 스케일 (기본값: 1.0)")
    parser.add_argument("--motion_scale", type=float, default=1.0,
                        help="motion_adapter 스케일 (기본값: 1.0)")

    # 생성 파라미터
    parser.add_argument("--guidance_scale",    type=float, default=None)
    parser.add_argument("--use_dynamic_cfg",   action="store_true", default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--fps",               type=int,   default=None)
    parser.add_argument("--num_frames",        type=int,   default=None)
    parser.add_argument("--height",            type=int,   default=None)
    parser.add_argument("--width",             type=int,   default=None)

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    lora_dir = os.path.abspath(args.lora_dir)
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"lora_dir 없음: {lora_dir}")

    # ── train_args.json으로 학습 설정 자동 로드 ────────────────────────────────
    train_args = load_train_args(lora_dir)
    if train_args:
        print(f"[train_args] {lora_dir}/train_args.json 로드 완료")

    def resolve(arg_val, key, fallback):
        if arg_val is not None:
            return arg_val
        if train_args and train_args.get(key) is not None:
            return train_args[key]
        return fallback

    model_path      = resolve(args.model_path, "pretrained_model_name_or_path", None)
    guidance_scale  = resolve(args.guidance_scale,  "guidance_scale",  6.0)
    fps             = resolve(args.fps,             "fps",             8)
    num_frames      = resolve(args.num_frames,      "max_num_frames",  49)
    height          = resolve(args.height,          "height",          480)
    width           = resolve(args.width,           "width",           720)
    use_dynamic_cfg = resolve(args.use_dynamic_cfg, "use_dynamic_cfg", True)

    if model_path is None:
        raise ValueError("--model_path 또는 train_args.json에 pretrained_model_name_or_path 필요")

    # ── 어댑터 경로 확인 ───────────────────────────────────────────────────────
    id_path     = os.path.join(lora_dir, "id_adapter")
    motion_path = os.path.join(lora_dir, "motion_adapter")

    has_id     = os.path.isdir(id_path)
    has_motion = os.path.isdir(motion_path)
    hw_id      = has_id     and is_head_wise(id_path)
    hw_motion  = has_motion and is_head_wise(motion_path)

    if not has_id and not has_motion:
        raise FileNotFoundError(f"id_adapter/ 또는 motion_adapter/ 가 없음: {lora_dir}")

    print("\n=== 어댑터 구조 ===")
    print(f"  id_adapter    : {'✅' if has_id else '❌'} "
          f"{'[Head-Wise]' if hw_id else '[PEFT]' if has_id else ''}")
    print(f"  motion_adapter: {'✅' if has_motion else '❌'} "
          f"{'[Head-Wise]' if hw_motion else '[PEFT]' if has_motion else ''}")

    # ── 출력 디렉토리 ──────────────────────────────────────────────────────────
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, f"head_wise_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    video_path       = os.path.join(output_dir, "video.mp4")
    first_frame_path = os.path.join(output_dir, "first_frame.png")
    args_json_path   = os.path.join(output_dir, "args.json")

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    print(f"\n모델 로드 중: {model_path}")
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.to("cuda")
    pipe.transformer.eval()

    # ── LoRA 로드 ──────────────────────────────────────────────────────────────
    id_injected     = {}
    motion_injected = {}

    if has_id:
        if hw_id:
            # Head-Wise LoRA
            print(f"[Head-Wise] id_adapter 로드: {id_path}")
            id_injected = load_head_wise_lora(
                pipe.transformer, id_path, device="cuda", dtype=torch.bfloat16
            )
            set_scaling_from_config(
                id_injected,
                os.path.join(id_path, "head_wise_config.json"),
                args.id_scale,
            )
        else:
            # 기존 PEFT LoRA fallback
            print(f"[PEFT] id_adapter 로드: {id_path}")
            pipe.load_lora_weights(id_path, adapter_name="id_adapter")

    if has_motion:
        if hw_motion:
            # Head-Wise LoRA
            print(f"[Head-Wise] motion_adapter 로드: {motion_path}")
            motion_injected = load_head_wise_lora(
                pipe.transformer, motion_path, device="cuda", dtype=torch.bfloat16
            )
            set_scaling_from_config(
                motion_injected,
                os.path.join(motion_path, "head_wise_config.json"),
                args.motion_scale,
            )
        else:
            # 기존 PEFT LoRA fallback
            print(f"[PEFT] motion_adapter 로드: {motion_path}")
            pipe.load_lora_weights(motion_path, adapter_name="motion_adapter")

    # PEFT 어댑터 스케일 설정 (PEFT fallback 시)
    peft_adapters = []
    peft_scales   = []
    if has_id and not hw_id:
        peft_adapters.append("id_adapter");     peft_scales.append(args.id_scale)
    if has_motion and not hw_motion:
        peft_adapters.append("motion_adapter"); peft_scales.append(args.motion_scale)
    if peft_adapters:
        pipe.set_adapters(peft_adapters, peft_scales)

    # Head-Wise 정보 출력
    if id_injected or motion_injected:
        print_lora_info(id_injected, motion_injected, args.id_scale, args.motion_scale)

    # ── Inference ──────────────────────────────────────────────────────────────
    print(f"\n=== Inference ===")
    print(f"  prompt              : {args.prompt}")
    print(f"  id_scale            : {args.id_scale}")
    print(f"  motion_scale        : {args.motion_scale}")
    print(f"  guidance_scale      : {guidance_scale}")
    print(f"  num_inference_steps : {args.num_inference_steps}")
    print(f"  use_dynamic_cfg     : {use_dynamic_cfg}")
    print(f"  {height}x{width}, {num_frames} frames, fps={fps}, seed={args.seed}")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.no_grad():
        frames = pipe(
            prompt=args.prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            generator=generator,
        ).frames[0]

    # ── 저장 ───────────────────────────────────────────────────────────────────
    export_to_video(frames, video_path, fps=fps)
    frames[0].save(first_frame_path)

    with open(args_json_path, "w") as f:
        json.dump({
            "lora_dir":           lora_dir,
            "prompt":             args.prompt,
            "model_path":         model_path,
            "id_scale":           args.id_scale,
            "motion_scale":       args.motion_scale,
            "guidance_scale":     guidance_scale,
            "num_inference_steps": args.num_inference_steps,
            "use_dynamic_cfg":    use_dynamic_cfg,
            "height":             height,
            "width":              width,
            "num_frames":         num_frames,
            "fps":                fps,
            "seed":               args.seed,
            "timestamp":          timestamp,
            "id_adapter_type":    "head_wise" if hw_id     else "peft" if has_id     else "none",
            "motion_adapter_type": "head_wise" if hw_motion else "peft" if has_motion else "none",
        }, f, indent=4, ensure_ascii=False)

    print(f"\n=== 저장 완료: {output_dir} ===")
    print(f"  video      : {video_path}")
    print(f"  first_frame: {first_frame_path}")
    print(f"  args       : {args_json_path}")


if __name__ == "__main__":
    main()
