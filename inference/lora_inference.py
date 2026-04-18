import os
import json
import torch
from datetime import datetime
from diffusers import CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
import argparse


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def print_adapter_info(pipe, adapter_names):
    from peft.tuners.tuners_utils import BaseTunerLayer
    print("\n=== Adapter 정보 ===")
    for adapter_name in adapter_names:
        count = 0
        scales = []
        for _, module in pipe.transformer.named_modules():
            if isinstance(module, BaseTunerLayer):
                if hasattr(module, "lora_A") and adapter_name in module.lora_A:
                    count += 1
                if hasattr(module, "scaling") and adapter_name in module.scaling:
                    s = module.scaling[adapter_name]
                    scales.append(s.item() if hasattr(s, "item") else s)
        scale_str = f"scale={scales[0]:.4f}" if scales else "scale=N/A"
        print(f"  {adapter_name}: lora_A layers={count}, {scale_str}")


def load_train_args(lora_dir):
    """train_args.json이 있으면 로드해서 반환, 없으면 None."""
    path = os.path.join(lora_dir, "train_args.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX dual-LoRA Inference")

    # 필수 경로
    parser.add_argument("--lora_dir", type=str, required=True,
                        help="학습 출력 디렉토리 (id_adapter/, motion_adapter/ 가 있는 폴더)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="생성할 영상의 텍스트 프롬프트")
    parser.add_argument("--results_dir", type=str, default="inference/results",
                        help="결과 저장 루트 디렉토리 (기본값: inference/results)")

    # 모델 경로 (train_args.json에서 자동 추론 가능)
    parser.add_argument("--model_path", type=str, default=None,
                        help="CogVideoX 모델 경로. 미지정 시 train_args.json에서 자동 로드")

    # Adapter 스케일
    parser.add_argument("--id_scale",     type=float, default=1.0,
                        help="id_adapter LoRA 스케일 (기본값: 1.0)")
    parser.add_argument("--motion_scale", type=float, default=1.0,
                        help="motion_adapter LoRA 스케일 (기본값: 1.0)")

    # 생성 파라미터 (미지정 시 train_args.json 값 사용)
    parser.add_argument("--guidance_scale",  type=float, default=None)
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=None)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--fps",             type=int,   default=None)
    parser.add_argument("--num_frames",      type=int,   default=None)
    parser.add_argument("--height",          type=int,   default=None)
    parser.add_argument("--width",           type=int,   default=None)

    return parser.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()

    lora_dir = os.path.abspath(args.lora_dir)
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(f"lora_dir을 찾을 수 없습니다: {lora_dir}")

    # ── train_args.json으로 학습 설정 로드 ────────
    train_args = load_train_args(lora_dir)
    if train_args:
        print(f"train_args.json 로드: {lora_dir}/train_args.json")

    def resolve(arg_val, key, fallback):
        """CLI 인자 > train_args.json > fallback 순으로 값 결정."""
        if arg_val is not None:
            return arg_val
        if train_args and key in train_args and train_args[key] is not None:
            return train_args[key]
        return fallback

    model_path    = resolve(args.model_path,    "pretrained_model_name_or_path", None)
    guidance_scale = resolve(args.guidance_scale, "guidance_scale", 6.0)
    fps           = resolve(args.fps,           "fps",           8)
    num_frames    = resolve(args.num_frames,    "max_num_frames", 49)
    height        = resolve(args.height,        "height",        480)
    width         = resolve(args.width,         "width",         720)
    use_dynamic_cfg = resolve(args.use_dynamic_cfg, "use_dynamic_cfg", True)

    if model_path is None:
        raise ValueError("--model_path 또는 train_args.json의 pretrained_model_name_or_path가 필요합니다.")

    # ── LoRA 파일 경로 확인 ───────────────────────
    id_adapter_path     = os.path.join(lora_dir, "id_adapter")
    motion_adapter_path = os.path.join(lora_dir, "motion_adapter")

    has_id     = os.path.isdir(id_adapter_path)
    has_motion = os.path.isdir(motion_adapter_path)

    if not has_id and not has_motion:
        raise FileNotFoundError(
            f"id_adapter/ 또는 motion_adapter/ 가 없습니다: {lora_dir}"
        )

    print(f"\n=== LoRA 구조 ===")
    print(f"  id_adapter    : {'✅ ' + id_adapter_path if has_id else '❌ 없음'}")
    print(f"  motion_adapter: {'✅ ' + motion_adapter_path if has_motion else '❌ 없음'}")

    # ── 타임스탬프 출력 디렉토리 생성 ─────────────
    # results/{YYYYMMDD_HHMMSS}/
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    video_path       = os.path.join(output_dir, "video.mp4")
    first_frame_path = os.path.join(output_dir, "video_first_frame.png")
    args_json_path   = os.path.join(output_dir, "args.json")

    # ── Generator ────────────────────────────────
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    # ── Pipeline 로드 ─────────────────────────────
    print(f"\n모델 로드 중: {model_path}")
    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()
    pipe.to("cuda")

    # ── LoRA 로드 ─────────────────────────────────
    active_adapters = []
    active_scales   = []

    if has_id:
        print(f"id_adapter 로드 중: {id_adapter_path}")
        pipe.load_lora_weights(id_adapter_path, adapter_name="id_adapter")
        active_adapters.append("id_adapter")
        active_scales.append(args.id_scale)

    if has_motion:
        print(f"motion_adapter 로드 중: {motion_adapter_path}")
        pipe.load_lora_weights(motion_adapter_path, adapter_name="motion_adapter")
        active_adapters.append("motion_adapter")
        active_scales.append(args.motion_scale)

    # ── Adapter 스케일 설정 ───────────────────────
    pipe.set_adapters(active_adapters, active_scales)
    print_adapter_info(pipe, active_adapters)

    # ── Inference ────────────────────────────────
    print(f"\n=== Inference 시작 ===")
    print(f"  prompt         : {args.prompt}")
    print(f"  id_scale       : {args.id_scale}")
    print(f"  motion_scale   : {args.motion_scale}")
    print(f"  guidance_scale : {guidance_scale}")
    print(f"  use_dynamic_cfg: {use_dynamic_cfg}")
    print(f"  height x width : {height} x {width}")
    print(f"  num_frames     : {num_frames}")
    print(f"  fps            : {fps}")
    print(f"  seed           : {args.seed}")
    print(f"  output_dir     : {output_dir}")

    frames = pipe(
        prompt=args.prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        use_dynamic_cfg=use_dynamic_cfg,
        generator=generator,
    ).frames[0]

    # ── 저장 ─────────────────────────────────────
    export_to_video(frames, video_path, fps=fps)
    frames[0].save(first_frame_path)

    # args.json 저장
    args_dict = {
        "lora_dir":        lora_dir,
        "prompt":          args.prompt,
        "model_path":      model_path,
        "id_scale":        args.id_scale,
        "motion_scale":    args.motion_scale,
        "guidance_scale":  guidance_scale,
        "use_dynamic_cfg": use_dynamic_cfg,
        "height":          height,
        "width":           width,
        "num_frames":      num_frames,
        "fps":             fps,
        "seed":            args.seed,
        "timestamp":       timestamp,
    }
    with open(args_json_path, "w") as f:
        json.dump(args_dict, f, indent=4, ensure_ascii=False)

    print(f"\n=== 저장 완료: {output_dir} ===")
    print(f"  video          : {video_path}")
    print(f"  first_frame    : {first_frame_path}")
    print(f"  args           : {args_json_path}")


if __name__ == "__main__":
    main()
