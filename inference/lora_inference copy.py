import os
import torch
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
import argparse


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def is_adapter_empty(pipe, adapter_name):
    count = 0
    for _, module in pipe.transformer.named_modules():
        if hasattr(module, "lora_A") and adapter_name in module.lora_A:
            count += 1
        if hasattr(module, "lora_B") and adapter_name in module.lora_B:
            count += 1
    return count == 0, count


def print_adapter_info(pipe, adapter_names):
    from peft.tuners.tuners_utils import BaseTunerLayer
    print("\n=== Adapter 정보 ===")
    for adapter_name in adapter_names:
        empty, count = is_adapter_empty(pipe, adapter_name)
        scales = []
        for _, module in pipe.transformer.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "scaling") and adapter_name in module.scaling:
                    s = module.scaling[adapter_name]
                    scales.append(s.item() if hasattr(s, "item") else s)
        scale_str = f"scale={scales[0]:.2f}" if scales else "scale=N/A"
        print(f"  {adapter_name}: layers={count}, {scale_str}")


def split_state_dict(full_state_dict):
    """
    저장된 state_dict를 id / motion / shared adapter별로 분리.

    train_lora_timestep.py 저장 순서: {**motion, **id, **shared}
    → 동일 key는 나중에 쓴 adapter(id, shared)가 최종 저장됨.
    → 실효 매핑:
        to_k, to_q               : id_adapter (id가 motion 덮어씀)
        to_v, to_out.0, ff.net.2 : motion_adapter (motion이 먼저, but id overwrites → id의 미학습 weight)
        ff.net.0.proj            : shared_adapter (shared가 최종)

    train_lora.py 저장 순서도 동일하나 target_modules가 달라 충돌 없음:
        id_adapter    → to_k, to_q
        motion_adapter→ to_v, to_out.0, ff.net.2.proj
        shared_adapter→ ff.net.0.proj
    """
    id_keys     = {k: v for k, v in full_state_dict.items()
                   if any(t in k for t in ["to_k", "to_q"])
                   and "ff.net" not in k}

    motion_keys = {k: v for k, v in full_state_dict.items()
                   if any(t in k for t in ["to_v", "to_out.0", "ff.net.2.proj"])}

    shared_keys = {k: v for k, v in full_state_dict.items()
                   if "ff.net.0.proj" in k}

    return id_keys, motion_keys, shared_keys


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX LoRA Inference")

    # 필수 경로
    parser.add_argument("--lora_path", type=str, required=True,
                        help="학습된 LoRA 경로 (디렉토리 또는 .safetensors 파일)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="생성할 영상의 텍스트 프롬프트")
    parser.add_argument("--output_path", type=str, default="output.mp4",
                        help="저장할 영상 경로 (기본값: output.mp4)")

    # 모델 경로
    parser.add_argument("--model_path", type=str,
                        default="/home/sbjeon/workspace/vc/DualReal/CogVideoX-5b",
                        help="CogVideoX 모델 경로")

    # Adapter 스케일
    parser.add_argument("--id_scale",     type=float, default=1.0,
                        help="id_adapter 스케일 (기본값: 1.0)")
    parser.add_argument("--motion_scale", type=float, default=1.0,
                        help="motion_adapter 스케일 (기본값: 1.0)")
    parser.add_argument("--shared_scale", type=float, default=1.0,
                        help="shared_adapter 스케일 (기본값: 1.0)")

    # 생성 파라미터
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--fps",            type=int,   default=8)
    parser.add_argument("--num_frames",     type=int,   default=49)

    return parser.parse_args()


def main():
    args = parse_args()

    # ── 출력 디렉토리 생성 ────────────────────────
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    # ── Generator ────────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ── Pipeline 로드 ─────────────────────────────
    print(f"모델 로드 중: {args.model_path}")
    pipe = CogVideoXPipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    # ── LoRA 로드 & Adapter 분리 ──────────────────
    # 절대 경로 변환 + 디렉토리인 경우 파일명 자동 추가
    lora_path = os.path.abspath(args.lora_path)
    if os.path.isdir(lora_path):
        lora_path = os.path.join(lora_path, "pytorch_lora_weights.safetensors")
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA 파일을 찾을 수 없습니다: {lora_path}")

    print(f"LoRA 로드 중: {lora_path}")
    full_state_dict = CogVideoXPipeline.lora_state_dict(lora_path)
    id_keys, motion_keys, shared_keys = split_state_dict(full_state_dict)

    print(f"  id_adapter    keys: {len(id_keys)}")
    print(f"  motion_adapter keys: {len(motion_keys)}")
    print(f"  shared_adapter keys: {len(shared_keys)}")

    pipe.load_lora_weights(id_keys,     adapter_name="id_adapter")
    pipe.load_lora_weights(motion_keys, adapter_name="motion_adapter")
    pipe.load_lora_weights(shared_keys, adapter_name="shared_adapter")

    # ── Adapter 스케일 설정 ───────────────────────
    pipe.set_adapters(
        ["id_adapter", "motion_adapter", "shared_adapter"],
        [args.id_scale, args.motion_scale, args.shared_scale],
    )
    print_adapter_info(pipe, ["id_adapter", "motion_adapter", "shared_adapter"])

    # ── Inference ────────────────────────────────
    print(f"\n=== Inference 시작 ===")
    print(f"  prompt      : {args.prompt}")
    print(f"  scales      : id={args.id_scale}, motion={args.motion_scale}, shared={args.shared_scale}")
    print(f"  guidance    : {args.guidance_scale}")
    print(f"  seed        : {args.seed}")
    print(f"  output      : {args.output_path}")

    frames = pipe(
        prompt=args.prompt,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        use_dynamic_cfg=False, ## Dynamic CFG 설정
        generator=generator,
    ).frames[0]

    # ── 저장 ─────────────────────────────────────
    export_to_video(frames, args.output_path, fps=args.fps)
    print(f"\n저장 완료: {args.output_path}")

    # 첫 프레임도 함께 저장
    first_frame_path = args.output_path.replace(".mp4", "_first_frame.png")
    frames[0].save(first_frame_path)
    print(f"첫 프레임 저장: {first_frame_path}")


if __name__ == "__main__":
    main()
