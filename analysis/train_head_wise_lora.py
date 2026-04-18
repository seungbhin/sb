"""
Head-Wise LoRA 학습 스크립트 (완전 독립 버전)

train_lora.py의 PEFT LoRA를 Head-Wise LoRA로 전면 대체.

  id_adapter     → spatial head의 Q, K  (temporal head Q/K 오염 방지)
  motion_adapter → temporal head의 V, Out (파라미터 91% 절감)

Usage:
    accelerate launch --config_file train/config/finetune_adapter_single.yaml \\
        analysis/train_head_wise_lora.py \\
        --pretrained_model_name_or_path $MODEL_PATH \\
        --instance_data_root_id   $ID_DATA_PATH \\
        --instance_data_root_motion $MOTION_DATA_PATH \\
        --output_dir ./output/train_lora/head_wise_test \\
        --id_rank 32 --id_lora_alpha 32.0 \\
        --motion_rank 32 --motion_lora_alpha 32.0

    (단축 인터페이스)
        --video_root_dir $VIDEO_PATH      →  instance_data_root_motion 에 매핑
        --id_image_dir   $ID_PATH         →  instance_data_root_id     에 매핑
"""

import argparse
import logging
import math
import os
import shutil
import sys
import json
import random
from copy import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from transformers import AutoTokenizer, T5EncoderModel
import transformers
import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXPipeline,
    CogVideoXTransformer3DModel,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import free_memory
from diffusers.utils import export_to_video, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

# Head-Wise LoRA 모듈
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.head_wise_lora import (
    HeadWiseLoRALinear,
    apply_both_adapters,
    save_both_adapters,
    load_head_wise_lora,
    print_parameter_comparison,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-Wise LoRA 학습 스크립트 for CogVideoX"
    )

    # ── LoRA 설정 ──────────────────────────────────────────────────────────────
    parser.add_argument("--id_rank",           type=int,   default=32)
    parser.add_argument("--id_lora_alpha",     type=float, default=32.0)
    parser.add_argument("--motion_rank",       type=int,   default=32)
    parser.add_argument("--motion_lora_alpha", type=float, default=32.0)

    # ── 모델 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision",  type=str, default=None)
    parser.add_argument("--variant",   type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)

    # ── 데이터셋 (train_lora.py 호환 + 단축 별칭) ─────────────────────────────
    parser.add_argument("--instance_data_root_id",     type=str, default=None,
                        help="id_adapter 학습 데이터 폴더")
    parser.add_argument("--instance_data_root_motion", type=str, default=None,
                        help="motion_adapter 학습 데이터 폴더")
    # 단축 별칭
    parser.add_argument("--id_image_dir",   type=str, default=None,
                        help="instance_data_root_id 별칭")
    parser.add_argument("--video_root_dir", type=str, default=None,
                        help="instance_data_root_motion 별칭")

    parser.add_argument("--video_column_id",      type=str, default="video")
    parser.add_argument("--video_column_motion",  type=str, default="video")
    parser.add_argument("--caption_column_id",    type=str, default="text")
    parser.add_argument("--caption_column_motion",type=str, default="text")
    parser.add_argument("--dataset_name",         type=str, default=None)
    parser.add_argument("--dataset_config_name",  type=str, default=None)
    parser.add_argument("--id_token",             type=str, default=None)
    parser.add_argument("--dataloader_num_workers",type=int, default=0)

    # ── 비디오 형상 ────────────────────────────────────────────────────────────
    parser.add_argument("--height",          type=int, default=480)
    parser.add_argument("--width",           type=int, default=720)
    parser.add_argument("--fps",             type=int, default=8)
    parser.add_argument("--max_num_frames",  type=int, default=49)
    parser.add_argument("--skip_frames_start", type=int, default=0)
    parser.add_argument("--skip_frames_end",   type=int, default=0)

    # ── 학습 ──────────────────────────────────────────────────────────────────
    parser.add_argument("--seed",               type=int,   default=None)
    parser.add_argument("--mixed_precision",    type=str,   default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--output_dir",         type=str,   default="head_wise_lora_output")
    parser.add_argument("--train_batch_size",   type=int,   default=1)
    parser.add_argument("--train_batch_size_other", type=int, default=1)
    parser.add_argument("--num_train_epochs",   type=int,   default=1)
    parser.add_argument("--max_train_steps",    type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--motion_ratio",       type=float, default=0.5,
                        help="각 step에서 motion 배치를 사용할 확률")

    # ── 최적화 ────────────────────────────────────────────────────────────────
    parser.add_argument("--learning_rate",     type=float, default=1e-4)
    parser.add_argument("--scale_lr",          action="store_true", default=False)
    parser.add_argument("--lr_scheduler",      type=str,   default="cosine_with_restarts")
    parser.add_argument("--lr_warmup_steps",   type=int,   default=200)
    parser.add_argument("--lr_num_cycles",     type=int,   default=1)
    parser.add_argument("--lr_power",          type=float, default=1.0)
    parser.add_argument("--optimizer",         type=lambda s: s.lower(),
                        default="adamw", choices=["adam", "adamw", "prodigy"])
    parser.add_argument("--use_8bit_adam",     action="store_true")
    parser.add_argument("--adam_beta1",        type=float, default=0.9)
    parser.add_argument("--adam_beta2",        type=float, default=0.95)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon",      type=float, default=1e-8)
    parser.add_argument("--max_grad_norm",     type=float, default=1.0)
    parser.add_argument("--prodigy_beta3",     type=float, default=None)
    parser.add_argument("--prodigy_decouple",  action="store_true")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true")
    parser.add_argument("--prodigy_safeguard_warmup",    action="store_true")

    # ── 체크포인트 ────────────────────────────────────────────────────────────
    parser.add_argument("--checkpointing_steps",    type=int, default=500)
    parser.add_argument("--checkpoints_total_limit",type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # ── VAE ───────────────────────────────────────────────────────────────────
    parser.add_argument("--enable_slicing", action="store_true", default=False)
    parser.add_argument("--enable_tiling",  action="store_true", default=False)

    # ── 검증 / 로깅 ───────────────────────────────────────────────────────────
    parser.add_argument("--tracker_name",          type=str, default=None)
    parser.add_argument("--report_to",             type=str, default="wandb")
    parser.add_argument("--logging_dir",           type=str, default="logs")
    parser.add_argument("--allow_tf32",            action="store_true")
    parser.add_argument("--hub_token",             type=str, default=None)
    parser.add_argument("--hub_model_id",          type=str, default=None)
    parser.add_argument("--num_validation_videos", type=int, default=1)
    parser.add_argument("--guidance_scale",        type=float, default=6.0)
    parser.add_argument("--use_dynamic_cfg",       action="store_true", default=True)
    parser.add_argument("--inference_steps",       type=int, default=500)
    parser.add_argument("--inf_eva_prompts",       type=str, default=None)
    parser.add_argument("--validation_prompt_separator", type=str, default=":::")

    args = parser.parse_args()

    # 단축 별칭 처리
    if args.id_image_dir and not args.instance_data_root_id:
        args.instance_data_root_id = args.id_image_dir
    if args.video_root_dir and not args.instance_data_root_motion:
        args.instance_data_root_motion = args.video_root_dir

    # 데이터 경로 검증
    if args.instance_data_root_id is None:
        parser.error("--instance_data_root_id (또는 --id_image_dir) 가 필요합니다.")
    if args.instance_data_root_motion is None:
        parser.error("--instance_data_root_motion (또는 --video_root_dir) 가 필요합니다.")

    return args


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (train_lora.py와 동일)
# ─────────────────────────────────────────────────────────────────────────────

class VideoDataset_for_id(Dataset):
    def __init__(
        self,
        instance_data_root=None, dataset_name=None, dataset_config_name=None,
        caption_column="text", video_column="video",
        height=480, width=720, fps=8, max_num_frames=49,
        skip_frames_start=0, skip_frames_end=0,
        cache_dir=None, id_token=None,
    ):
        super().__init__()
        self.instance_data_root   = Path(instance_data_root) if instance_data_root else None
        self.dataset_name         = dataset_name
        self.dataset_config_name  = dataset_config_name
        self.caption_column       = caption_column
        self.video_column         = video_column
        self.height, self.width   = height, width
        self.fps                  = fps
        self.max_num_frames       = max_num_frames
        self.skip_frames_start    = skip_frames_start
        self.skip_frames_end      = skip_frames_end
        self.cache_dir            = cache_dir
        self.id_token             = id_token or ""

        if dataset_name:
            self.instance_prompts, self.instance_video_paths = self._load_from_hub()
        else:
            self.instance_prompts, self.instance_video_paths = self._load_from_local()

        self.num_instance_videos = len(self.instance_video_paths)
        assert self.num_instance_videos == len(self.instance_prompts)

        self.instance_videos, self.instance_vit_frames = self._preprocess_data()

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            "instance_video":  self.instance_videos[index],
            "instance_vit_frames": self.instance_vit_frames[index],
        }

    def _load_from_hub(self):
        from datasets import load_dataset
        dataset = load_dataset(self.dataset_name, self.dataset_config_name, cache_dir=self.cache_dir)
        return dataset["train"][self.caption_column], [
            Path(self.instance_data_root, p) for p in dataset["train"][self.video_column]
        ]

    def _load_from_local(self):
        prompt_path = self.instance_data_root / self.caption_column
        video_path  = self.instance_data_root / self.video_column
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
        with open(video_path, "r", encoding="utf-8") as f:
            videos  = [self.instance_data_root / l.strip() for l in f if l.strip()]
        return prompts, videos

    def _preprocess_data(self):
        import decord
        decord.bridge.set_bridge("torch")
        to_tensor = transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        videos, vit_frames = [], []
        for filename in self.instance_video_paths:
            vr     = decord.VideoReader(uri=str(filename), width=self.width,  height=self.height)
            vr_ref = decord.VideoReader(uri=str(filename), width=224,          height=224)
            n = len(vr)
            s = min(self.skip_frames_start, n)
            e = max(0, n - self.skip_frames_end)
            idxs = list(range(s, e)) if e > s else [s]
            idxs = idxs[:self.max_num_frames]
            frames     = vr.get_batch(idxs).float()
            ref_frames = vr_ref.get_batch(idxs).float()

            # (4k+1) 맞추기
            rem = (3 + (len(frames) % 4)) % 4
            if rem:
                frames = frames[:-rem]
            assert (len(frames) - 1) % 4 == 0

            frames = torch.stack([to_tensor(f) for f in frames], dim=0)
            videos.append(frames.permute(0, 3, 1, 2).contiguous())

            mid = copy(ref_frames[0]).float()
            vit_frames.append(to_tensor(mid).permute(2, 0, 1).contiguous())
        return videos, vit_frames


class VideoDataset_for_motion(Dataset):
    def __init__(
        self,
        instance_data_root=None, dataset_name=None, dataset_config_name=None,
        caption_column="text", video_column="video",
        height=480, width=720, fps=8, max_num_frames=49,
        skip_frames_start=0, skip_frames_end=0,
        cache_dir=None, id_token=None,
    ):
        super().__init__()
        self.instance_data_root   = Path(instance_data_root) if instance_data_root else None
        self.dataset_name         = dataset_name
        self.dataset_config_name  = dataset_config_name
        self.caption_column       = caption_column
        self.video_column         = video_column
        self.height, self.width   = height, width
        self.fps                  = fps
        self.max_num_frames       = max_num_frames
        self.skip_frames_start    = skip_frames_start
        self.skip_frames_end      = skip_frames_end
        self.cache_dir            = cache_dir
        self.id_token             = id_token or ""

        if dataset_name:
            self.instance_prompts, self.instance_video_paths = self._load_from_hub()
        else:
            self.instance_prompts, self.instance_video_paths = self._load_from_local()

        self.num_instance_videos = len(self.instance_video_paths)
        assert self.num_instance_videos == len(self.instance_prompts)

        self.instance_videos, self.instance_vit_frames = self._preprocess_data()

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            "instance_video":  self.instance_videos[index],
            "instance_vit_frames": self.instance_vit_frames[index],
        }

    def _load_from_hub(self):
        from datasets import load_dataset
        dataset = load_dataset(self.dataset_name, self.dataset_config_name, cache_dir=self.cache_dir)
        return dataset["train"][self.caption_column], [
            Path(self.instance_data_root, p) for p in dataset["train"][self.video_column]
        ]

    def _load_from_local(self):
        prompt_path = self.instance_data_root / self.caption_column
        video_path  = self.instance_data_root / self.video_column
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompts = [l.strip() for l in f if l.strip()]
        with open(video_path, "r", encoding="utf-8") as f:
            videos  = [self.instance_data_root / l.strip() for l in f if l.strip()]
        return prompts, videos

    def _preprocess_data(self):
        import decord
        decord.bridge.set_bridge("torch")
        to_tensor = transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)
        videos, vit_frames = [], []
        for filename in self.instance_video_paths:
            vr     = decord.VideoReader(uri=str(filename), width=self.width,  height=self.height)
            vr_ref = decord.VideoReader(uri=str(filename), width=224,          height=224)
            n = len(vr)
            s = min(self.skip_frames_start, n)
            e = max(0, n - self.skip_frames_end)
            idxs = list(range(s, e)) if e > s else [s]
            idxs = idxs[:self.max_num_frames]
            frames     = vr.get_batch(idxs).float()
            ref_frames = vr_ref.get_batch(idxs).float()

            rem = (3 + (len(frames) % 4)) % 4
            if rem:
                frames = frames[:-rem]
            assert (len(frames) - 1) % 4 == 0

            frames = torch.stack([to_tensor(f) for f in frames], dim=0)
            videos.append(frames.permute(0, 3, 1, 2).contiguous())

            ref_idx = np.random.randint(0, len(ref_frames))
            mid = copy(ref_frames[ref_idx]).float()
            vit_frames.append(to_tensor(mid).permute(2, 0, 1).contiguous())
        return videos, vit_frames


# ─────────────────────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _get_t5_prompt_embeds(tokenizer, text_encoder, prompt, num_videos_per_prompt=1,
                          max_sequence_length=226, device=None, dtype=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs = tokenizer(
        prompt, padding="max_length", max_length=max_sequence_length,
        truncation=True, add_special_tokens=True, return_tensors="pt",
    )
    embeds = text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=dtype, device=device)
    _, seq_len, _ = embeds.shape
    embeds = embeds.repeat(1, num_videos_per_prompt, 1)
    return embeds.view(len(prompt) * num_videos_per_prompt, seq_len, -1)


def compute_prompt_embeddings(tokenizer, text_encoder, prompt, max_sequence_length,
                               device, dtype, requires_grad=False):
    fn = _get_t5_prompt_embeds
    ctx = torch.no_grad() if not requires_grad else torch.enable_grad()
    with ctx:
        return fn(tokenizer, text_encoder, prompt,
                  max_sequence_length=max_sequence_length, device=device, dtype=dtype)


def prepare_rotary_positional_embeddings(
    height, width, num_frames,
    vae_scale_factor_spatial=8, patch_size=2, attention_head_dim=64,
    device=None, base_height=480, base_width=720,
):
    grid_h = height // (vae_scale_factor_spatial * patch_size)
    grid_w = width  // (vae_scale_factor_spatial * patch_size)
    base_h = base_height // (vae_scale_factor_spatial * patch_size)
    base_w = base_width  // (vae_scale_factor_spatial * patch_size)
    grid_crops_coords = get_resize_crop_region_for_grid((grid_h, grid_w), base_w, base_h)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_h, grid_w),
        temporal_size=num_frames,
    )
    return freqs_cos.to(device=device), freqs_sin.to(device=device)


def get_optimizer(args, params):
    if args.optimizer == "adamw":
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            cls = bnb.optim.AdamW8bit
        else:
            cls = torch.optim.AdamW
        return cls(params, betas=(args.adam_beta1, args.adam_beta2),
                   eps=args.adam_epsilon, weight_decay=args.adam_weight_decay)
    elif args.optimizer == "adam":
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            cls = bnb.optim.Adam8bit
        else:
            cls = torch.optim.Adam
        return cls(params, betas=(args.adam_beta1, args.adam_beta2),
                   eps=args.adam_epsilon, weight_decay=args.adam_weight_decay)
    elif args.optimizer == "prodigy":
        import prodigyopt
        return prodigyopt.Prodigy(
            params, lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3, weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon, decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )


def log_validation(pipe, args, accelerator, pipeline_args, step, is_final=False, save_dir=None):
    logger.info(f"Running validation... prompt: {pipeline_args['prompt']}")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(accelerator.device)
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
    phase = "test" if is_final else "validation"
    out_dir = save_dir or args.output_dir
    slug = pipeline_args["prompt"][:25].replace(" ", "_").replace("/", "_")
    videos, filenames = [], []
    for i in range(args.num_validation_videos):
        with torch.no_grad():
            pt_imgs = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
        pt_imgs   = torch.stack([pt_imgs[j] for j in range(pt_imgs.shape[0])])
        image_np  = VaeImageProcessor.pt_to_numpy(pt_imgs)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)
        videos.append(image_pil)
        fname = os.path.join(out_dir, f"{phase}_step{step}_video{i}_{slug}.mp4")
        export_to_video(image_pil, fname, fps=args.fps)
        filenames.append(fname)
        logger.info(f"Saved: {fname}")
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({phase: [wandb.Video(f) for f in filenames]}, step=step)
    return videos


# ─────────────────────────────────────────────────────────────────────────────
# Head-Wise LoRA 어댑터 전환 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def set_active_adapter(adapter_name: str,
                       id_injected: Dict,
                       motion_injected: Dict):
    """
    PEFT의 transformer.set_adapters([name]) 대체.
    지정된 adapter의 lora_A, lora_B만 requires_grad=True.
    두 어댑터는 서로 다른 projection을 담당하므로,
    forward는 항상 양쪽 모두 실행(간섭 없음),
    gradient는 지정된 쪽만 흐름.
    """
    use_id     = (adapter_name == "id_adapter")
    use_motion = (adapter_name == "motion_adapter")
    for m in id_injected.values():
        m.lora_A.requires_grad_(use_id)
        m.lora_B.requires_grad_(use_id)
    for m in motion_injected.values():
        m.lora_A.requires_grad_(use_motion)
        m.lora_B.requires_grad_(use_motion)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    _output_dir = args.output_dir  # 예외 시 정리용

    if args.report_to == "wandb" and args.hub_token:
        raise ValueError("--report_to=wandb 와 --hub_token 을 동시에 사용할 수 없습니다.")

    logging_dir = Path(args.output_dir) / args.logging_dir
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    load_dtype = (
        torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    )
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        torch_dtype=load_dtype, revision=args.revision, variant=args.variant,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant,
    )
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    if args.enable_slicing: vae.enable_slicing()
    if args.enable_tiling:  vae.enable_tiling()

    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    # weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # ── Head-Wise LoRA 주입 ──────────────────────────────────────────────────
    if accelerator.is_main_process:
        print("\n[파라미터 비교 - Head-Wise vs Standard]")
        print_parameter_comparison(
            id_rank=args.id_rank,
            motion_rank=args.motion_rank,
            std_rank=max(args.id_rank, args.motion_rank),
        )

    id_injected, motion_injected = apply_both_adapters(
        transformer,
        id_rank=args.id_rank,
        id_alpha=args.id_lora_alpha,
        motion_rank=args.motion_rank,
        motion_alpha=args.motion_lora_alpha,
    )

    # fp16 혼합 정밀도일 때 LoRA 파라미터를 float32로 업캐스트
    if args.mixed_precision == "fp16":
        for m in list(id_injected.values()) + list(motion_injected.values()):
            m.lora_A.data = m.lora_A.data.to(torch.float32)
            m.lora_B.data = m.lora_B.data.to(torch.float32)

    # ── unwrap 헬퍼 ───────────────────────────────────────────────────────────
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        return model._orig_mod if is_compiled_module(model) else model

    # ── save hook (PEFT 대신 head_wise_lora.save_both_adapters 사용) ──────────
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    save_both_adapters(
                        id_injected=id_injected,
                        motion_injected=motion_injected,
                        output_dir=output_dir,
                        id_rank=args.id_rank,
                        id_alpha=args.id_lora_alpha,
                        motion_rank=args.motion_rank,
                        motion_alpha=args.motion_lora_alpha,
                    )
                weights.pop()

    # ── load hook (체크포인트 복원) ───────────────────────────────────────────
    def load_model_hook(models, input_dir):
        while models:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                for subdir in ["id_adapter", "motion_adapter"]:
                    path = os.path.join(input_dir, subdir)
                    cfg  = os.path.join(path, "head_wise_config.json")
                    if os.path.exists(cfg):
                        load_head_wise_lora(
                            unwrap_model(model), path,
                            device=str(accelerator.device),
                            dtype=weight_dtype,
                        )

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # 초기 param_group은 전체 LoRA 파라미터로 구성
    # (per-step에서 optimizer.param_groups를 재구성하므로 여기서는 dummy도 무방)
    all_lora_params = [
        p for m in list(id_injected.values()) + list(motion_injected.values())
        for p in [m.lora_A, m.lora_B]
    ]
    optimizer = get_optimizer(args, [{"params": all_lora_params, "lr": args.learning_rate}])

    # ── Dataset & DataLoader ─────────────────────────────────────────────────
    train_dataset_id = VideoDataset_for_id(
        instance_data_root=args.instance_data_root_id,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column_id,
        video_column=args.video_column_id,
        height=args.height, width=args.width, fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )
    train_dataset_motion = VideoDataset_for_motion(
        instance_data_root=args.instance_data_root_motion,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column_motion,
        video_column=args.video_column_motion,
        height=args.height, width=args.width, fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )

    # VAE 인코딩 (미리 캐싱)
    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0)
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        return vae.encode(video).latent_dist

    train_dataset_id.instance_videos     = [encode_video(v) for v in train_dataset_id.instance_videos]
    train_dataset_motion.instance_videos = [encode_video(v) for v in train_dataset_motion.instance_videos]

    def collate_fn(examples):
        videos  = [ex["instance_video"].sample() * vae.config.scaling_factor for ex in examples]
        prompts = [ex["instance_prompt"] for ex in examples]
        videos  = torch.cat(videos).permute(0, 2, 1, 3, 4)
        videos  = videos.to(memory_format=torch.contiguous_format).float()
        return {"videos": videos, "prompts": prompts}

    train_dataloader_id = DataLoader(
        train_dataset_id,
        batch_size=args.train_batch_size_other,
        shuffle=True, collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )
    train_dataloader_motion = DataLoader(
        train_dataset_motion,
        batch_size=args.train_batch_size,
        shuffle=True, collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
    )

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader_id) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # ── Accelerate prepare ───────────────────────────────────────────────────
    transformer, optimizer, train_dataloader_id, train_dataloader_motion, lr_scheduler = (
        accelerator.prepare(
            transformer, optimizer, train_dataloader_id, train_dataloader_motion, lr_scheduler
        )
    )

    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-head-wise-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # ── Prompt & RoPE 캐싱 ───────────────────────────────────────────────────
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    prompt_embeds_cache: Dict[str, torch.Tensor] = {}
    all_prompts = (
        [p for batch in train_dataloader_id     for p in batch["prompts"]]
        + [p for batch in train_dataloader_motion for p in batch["prompts"]]
    )
    for prompt in set(all_prompts):
        prompt_embeds_cache[prompt] = compute_prompt_embeddings(
            tokenizer, text_encoder, [prompt],
            model_config.max_text_seq_length,
            accelerator.device, weight_dtype,
        )

    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    latent_num_frames = train_dataset_id.instance_videos[0].sample().shape[1]
    cached_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=args.height, width=args.width,
            num_frames=latent_num_frames,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_size=model_config.patch_size,
            attention_head_dim=model_config.attention_head_dim,
            device=accelerator.device,
        )
        if model_config.use_rotary_positional_embeddings else None
    )

    # ── 학습 루프 ─────────────────────────────────────────────────────────────
    num_trainable = sum(p.numel() for p in all_lora_params)
    logger.info("***** Running training (Head-Wise LoRA) *****")
    logger.info(f"  id_adapter     : {len(id_injected)} 모듈 (spatial Q, K)")
    logger.info(f"  motion_adapter : {len(motion_injected)} 모듈 (temporal V, Out)")
    logger.info(f"  Num trainable parameters = {num_trainable:,}")
    logger.info(f"  Num examples (id+motion) = {len(train_dataset_id)+len(train_dataset_motion)}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0

    def next_or_restart(iterator, dataloader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(dataloader)
            return next(iterator), iterator

    # Resume from checkpoint
    initial_global_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1])
            )
            path = dirs[-1] if dirs else None

        if path is None:
            logger.info("체크포인트가 없으므로 처음부터 시작합니다.")
            args.resume_from_checkpoint = None
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = initial_global_step = int(path.split("-")[1])

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    iter_id     = iter(train_dataloader_id)
    iter_motion = iter(train_dataloader_motion)

    while initial_global_step < args.max_train_steps:
        transformer.train()

        batch_id,     iter_id     = next_or_restart(iter_id,     train_dataloader_id)
        batch_motion, iter_motion = next_or_restart(iter_motion, train_dataloader_motion)

        use_motion   = random.random() < args.motion_ratio
        batch        = batch_motion if use_motion else batch_id
        adapter_name = "motion_adapter" if use_motion else "id_adapter"

        # ── 활성 어댑터 설정 (requires_grad 토글) ─────────────────────────────
        set_active_adapter(adapter_name, id_injected, motion_injected)

        with accelerator.accumulate(transformer):
            model_input = batch["videos"].to(dtype=weight_dtype)
            prompts     = batch["prompts"]
            prompt_embeds = torch.cat([prompt_embeds_cache[p] for p in prompts], dim=0)

            noise    = torch.randn_like(model_input)
            B, F, C, height, width = model_input.shape
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (B,), device=model_input.device
            ).long()

            noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

            model_output = transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                image_rotary_emb=cached_rotary_emb,
                return_dict=False,
            )[0]

            if use_motion:
               B = model_output.shape[0]
               offset_noise = torch.randn(B, 1, 1, height, width, device=model_input.device, dtype=model_output.dtype)
               model_output = model_output + offset_noise

            model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

            alphas_cumprod = scheduler.alphas_cumprod[timesteps]
            weights = 1 / (1 - alphas_cumprod)
            while len(weights.shape) < len(model_pred.shape):
                weights = weights.unsqueeze(-1)

            loss = torch.mean(
                (weights * (model_pred - model_input) ** 2).reshape(B, -1), dim=1
            ).mean()
            accelerator.backward(loss)

            # optimizer param_group 재구성 (현재 requires_grad=True인 파라미터만)
            if accelerator.sync_gradients and accelerator.state.deepspeed_plugin is None:
                trainable_params = [p for p in transformer.parameters() if p.requires_grad]
                optimizer.param_groups = []
                optimizer.add_param_group({"params": trainable_params, "lr": args.learning_rate})

                accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            elif accelerator.sync_gradients and accelerator.state.deepspeed_plugin is not None:
                lr_scheduler.step()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                # 체크포인트 저장
                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1])
                        )
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            n_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            for ckpt in checkpoints[:n_remove]:
                                shutil.rmtree(os.path.join(args.output_dir, ckpt))
                                logger.info(f"Removed checkpoint: {ckpt}")

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved checkpoint: {save_path}")

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                "adapter": adapter_name}
        progress_bar.set_postfix(**logs)
        accelerator.log({"loss": logs["loss"], "lr": logs["lr"]}, step=global_step)

        initial_global_step += 1
        print(f"\033[1;31m training step: \033[0m{initial_global_step}  [{adapter_name}]")

    # ── 최종 저장 ─────────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer_unwrapped = unwrap_model(transformer)
        transformer_unwrapped.to(weight_dtype)

        # LoRA 가중치 저장
        save_both_adapters(
            id_injected=id_injected,
            motion_injected=motion_injected,
            output_dir=args.output_dir,
            id_rank=args.id_rank,
            id_alpha=args.id_lora_alpha,
            motion_rank=args.motion_rank,
            motion_alpha=args.motion_lora_alpha,
        )

        # 학습 인자 저장
        args_dict = vars(args).copy()
        args_dict["train_script"] = "analysis/train_head_wise_lora.py"
        args_dict["lora_type"]    = "head_wise"
        args_dict["adapter_config"] = {
            "id_adapter":     {"rank": args.id_rank,     "lora_alpha": args.id_lora_alpha,
                               "target": "spatial heads → Q, K"},
            "motion_adapter": {"rank": args.motion_rank, "lora_alpha": args.motion_lora_alpha,
                               "target": "temporal heads → V, Out"},
        }
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump(args_dict, f, indent=4, default=str)

        logger.info(f"Saved LoRA weights to {args.output_dir}")

        # ── 최종 validation inference ─────────────────────────────────────────
        del transformer_unwrapped
        free_memory()

        pipe = CogVideoXPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision, variant=args.variant,
            torch_dtype=weight_dtype,
        )
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
        if args.enable_slicing: pipe.vae.enable_slicing()
        if args.enable_tiling:  pipe.vae.enable_tiling()

        # Head-Wise LoRA 로드 (두 어댑터 모두)
        id_path     = os.path.join(args.output_dir, "id_adapter")
        motion_path = os.path.join(args.output_dir, "motion_adapter")
        if os.path.exists(os.path.join(id_path, "head_wise_config.json")):
            load_head_wise_lora(pipe.transformer, id_path,
                                device=str(accelerator.device), dtype=weight_dtype)
        if os.path.exists(os.path.join(motion_path, "head_wise_config.json")):
            load_head_wise_lora(pipe.transformer, motion_path,
                                device=str(accelerator.device), dtype=weight_dtype)

        if args.inf_eva_prompts and args.num_validation_videos > 0:
            import json as _json
            validation_prompts = _json.loads(args.inf_eva_prompts)
            for prompt in validation_prompts:
                pipeline_args = {
                    "prompt": prompt,
                    "guidance_scale":   args.guidance_scale,
                    "use_dynamic_cfg":  args.use_dynamic_cfg,
                    "height": args.height,
                    "width":  args.width,
                    "num_frames": args.max_num_frames,
                }
                log_validation(
                    pipe=pipe, args=args, accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    step=initial_global_step,
                    is_final=True,
                )
        del pipe
        free_memory()

    accelerator.end_training()

    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    _args = None
    try:
        main()
    except Exception:
        raise
