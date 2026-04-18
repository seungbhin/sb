# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
from copy import copy, deepcopy
from torchvision import transforms
import torch
import torchvision.transforms as TT
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
import transformers
import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler, CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import check_min_version, convert_unet_state_dict_to_peft, export_to_video, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

import random

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")
    ## lora 설정 (공통 기본값)
    # parser.add_argument(
    #     "--rank",
    #     type=int,
    #     default=8,
    #     help=("The dimension of the LoRA update matrices. Used as default for all adapters."),
    # )
    # parser.add_argument(
    #     "--lora_alpha",
    #     type=float,
    #     default=8,
    #     help=("The scaling factor to scale LoRA weight update. Used as default for all adapters."),
    # )
    ## adapter별 rank/alpha (미지정 시 --rank, --lora_alpha 값 사용)
    parser.add_argument("--id_rank",           type=int,   default=None)
    parser.add_argument("--id_lora_alpha",     type=float, default=None)
    parser.add_argument("--motion_rank",       type=int,   default=None)
    parser.add_argument("--motion_lora_alpha", type=float, default=None)
    # parser.add_argument("--shared_rank",       type=int,   default=None)
    # parser.add_argument("--shared_lora_alpha", type=float, default=None)

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    # Dataset information
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_root_id",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--instance_data_root_motion",
        type=str,
        default=None,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--video_column_id",
        type=str,
        default="video",
    )
    parser.add_argument(
        "--video_column_motion",
        type=str,
        default="video",
    )
    parser.add_argument(
        "--caption_column_id",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--caption_column_motion",
        type=str,
        default="text",
    )
    parser.add_argument(
        "--id_token", type=str, default=None, help="Identifier token appended to the start of each prompt if provided."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='bf16',
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="All input videos are resized to this height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=720,
        help="All input videos are resized to this width.",
    )
    parser.add_argument("--fps", type=int, default=8, help="All input videos will be used at this FPS.")
    parser.add_argument(
        "--max_num_frames", type=int, default=49, help="All input videos will be truncated to these many frames."
    )
    parser.add_argument(
        "--skip_frames_start",
        type=int,
        default=0,
        help="Number of frames to skip from the beginning of each input video. Useful if training data contains intro sequences.",
    )
    parser.add_argument(
        "--skip_frames_end",
        type=int,
        default=0,
        help="Number of frames to skip from the end of each input video. Useful if training data contains outro sequences.",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip videos horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--train_batch_size_other", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine_with_restarts",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=200, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adamw",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    # dualreal
    parser.add_argument(
        "--use_controller",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--hypernet_outputdim",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--close_grad_mask",
        type=bool,
        default=False,
    ) 
    parser.add_argument(
        "--motion_ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--spatial_adapter_alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--temporal_adapter_alpha",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--training_parameters",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_IdAdapter",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--use_MotionAdapter",
        type=bool,
        default=False,
    )


    parser.add_argument(
        "--inference_steps",
        type=int,
        default=500, #100000에서 변경 
    )

    parser.add_argument(
        "--inf_eva_prompts",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--spatial_adapter_hidden_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--temporal_adapter_hidden_dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--spatial_adapter_condition_dim",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--temporal_adapter_condition_dim",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--clip_pretrained",
        type=str,
    )
    
    parser.add_argument(
        "--p_image_zero",
        type=float,
        default="0",
    )
    return parser.parse_args()

class VideoDataset_for_id(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""

        if dataset_name is not None:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_hub()
        else:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        videos, vit_frames = self._preprocess_data()
        self.instance_videos = videos
        self.instance_vit_frames = vit_frames # [num_instance_videos, C, H, W]

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            "instance_video": self.instance_videos[index],
            "instance_vit_frames": self.instance_vit_frames[index],
        }

    def _load_dataset_from_hub(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                "local folder containing images only, specify --instance_data_root instead."
            )

        # Downloading and loading a dataset from the hub. See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            cache_dir=self.cache_dir,
        )
        column_names = dataset["train"].column_names

        if self.video_column is None:
            video_column = column_names[0]
            logger.info(f"`video_column` defaulting to {video_column}")
        else:
            video_column = self.video_column
            if video_column not in column_names:
                raise ValueError(
                    f"`--video_column` value '{video_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.caption_column is None:
            caption_column = column_names[1]
            logger.info(f"`caption_column` defaulting to {caption_column}")
        else:
            caption_column = self.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        instance_prompts = dataset["train"][caption_column]
        instance_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][video_column]]

        return instance_prompts, instance_videos

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return instance_prompts, instance_videos

    def _preprocess_data(self):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        decord.bridge.set_bridge("torch")

        videos = []
        vit_frames = []
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        for filename in self.instance_video_paths:
            video_reader = decord.VideoReader(uri=filename.as_posix(), width=self.width, height=self.height)
            video_reader_refimg = decord.VideoReader(uri=filename.as_posix(), width=224, height=224)
            video_num_frames = len(video_reader) # 1

            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
                ref_frame = video_reader_refimg.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
                ref_frame = video_reader_refimg.get_batch(list(range(start_frame, end_frame)))
            else:
                indices = list(range(start_frame, self.max_num_frames))
                frames = video_reader.get_batch(indices)
                ref_frame = video_reader_refimg.get_batch(indices)

            ref_idx = 0 # 只有一帧
            mid_frame = copy(ref_frame[ref_idx])
            mid_frame = mid_frame.float()
            vit_frame = train_transforms(mid_frame)
            
            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0
            
            # Training transforms
            frames = frames.float()
            frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
            videos.append(frames.permute(0, 3, 1, 2).contiguous())  # [F, C, H, W]
            vit_frames.append(vit_frame.permute(2, 0, 1).contiguous()) # [C, H, W]

        return videos, vit_frames

class VideoDataset_for_motion(Dataset):
    def __init__(
        self,
        instance_data_root: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_config_name: Optional[str] = None,
        caption_column: str = "text",
        video_column: str = "video",
        height: int = 480,
        width: int = 720,
        fps: int = 8,
        max_num_frames: int = 49,
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        cache_dir: Optional[str] = None,
        id_token: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.instance_data_root = Path(instance_data_root) if instance_data_root is not None else None
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.caption_column = caption_column
        self.video_column = video_column
        self.height = height
        self.width = width
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.cache_dir = cache_dir
        self.id_token = id_token or ""

        if dataset_name is not None:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_hub()
        else:
            self.instance_prompts, self.instance_video_paths = self._load_dataset_from_local_path()

        self.num_instance_videos = len(self.instance_video_paths)
        if self.num_instance_videos != len(self.instance_prompts):
            raise ValueError(
                f"Expected length of instance prompts and videos to be the same but found {len(self.instance_prompts)=} and {len(self.instance_video_paths)=}. Please ensure that the number of caption prompts and videos match in your dataset."
            )
        videos, vit_frames = self._preprocess_data()
        self.instance_videos = videos
        self.instance_vit_frames = vit_frames

    def __len__(self):
        return self.num_instance_videos

    def __getitem__(self, index):
        return {
            "instance_prompt": self.id_token + self.instance_prompts[index],
            "instance_video": self.instance_videos[index],
            "instance_vit_frames": self.instance_vit_frames[index],
        }

    def _load_dataset_from_hub(self):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "You are trying to load your data using the datasets library. If you wish to train using custom "
                "captions please install the datasets library: `pip install datasets`. If you wish to load a "
                "local folder containing images only, specify --instance_data_root instead."
            )

        # Downloading and loading a dataset from the hub. See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script
        dataset = load_dataset(
            self.dataset_name,
            self.dataset_config_name,
            cache_dir=self.cache_dir,
        )
        column_names = dataset["train"].column_names

        if self.video_column is None:
            video_column = column_names[0]
            logger.info(f"`video_column` defaulting to {video_column}")
        else:
            video_column = self.video_column
            if video_column not in column_names:
                raise ValueError(
                    f"`--video_column` value '{video_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        if self.caption_column is None:
            caption_column = column_names[1]
            logger.info(f"`caption_column` defaulting to {caption_column}")
        else:
            caption_column = self.caption_column
            if self.caption_column not in column_names:
                raise ValueError(
                    f"`--caption_column` value '{self.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
                )

        instance_prompts = dataset["train"][caption_column]
        instance_videos = [Path(self.instance_data_root, filepath) for filepath in dataset["train"][video_column]]

        return instance_prompts, instance_videos

    def _load_dataset_from_local_path(self):
        if not self.instance_data_root.exists():
            raise ValueError("Instance videos root folder does not exist")

        prompt_path = self.instance_data_root.joinpath(self.caption_column)
        video_path = self.instance_data_root.joinpath(self.video_column)

        if not prompt_path.exists() or not prompt_path.is_file():
            raise ValueError(
                "Expected `--caption_column` to be path to a file in `--instance_data_root` containing line-separated text prompts."
            )
        if not video_path.exists() or not video_path.is_file():
            raise ValueError(
                "Expected `--video_column` to be path to a file in `--instance_data_root` containing line-separated paths to video data in the same directory."
            )

        with open(prompt_path, "r", encoding="utf-8") as file:
            instance_prompts = [line.strip() for line in file.readlines() if len(line.strip()) > 0]
        with open(video_path, "r", encoding="utf-8") as file:
            instance_videos = [
                self.instance_data_root.joinpath(line.strip()) for line in file.readlines() if len(line.strip()) > 0
            ]

        if any(not path.is_file() for path in instance_videos):
            raise ValueError(
                "Expected '--video_column' to be a path to a file in `--instance_data_root` containing line-separated paths to video data but found atleast one path that is not a valid file."
            )

        return instance_prompts, instance_videos

    def _preprocess_data(self):
        try:
            import decord
        except ImportError:
            raise ImportError(
                "The `decord` package is required for loading the video dataset. Install with `pip install decord`"
            )

        decord.bridge.set_bridge("torch")

        videos = []
        vit_frames = []
        train_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

        for filename in self.instance_video_paths:
            video_reader = decord.VideoReader(uri=filename.as_posix(), width=self.width, height=self.height)
            video_reader_refimg = decord.VideoReader(uri=filename.as_posix(), width=224, height=224)
            video_num_frames = len(video_reader)

            start_frame = min(self.skip_frames_start, video_num_frames)
            end_frame = max(0, video_num_frames - self.skip_frames_end)
            if end_frame <= start_frame:
                frames = video_reader.get_batch([start_frame])
                ref_frame = video_reader_refimg.get_batch([start_frame])
            elif end_frame - start_frame <= self.max_num_frames:
                frames = video_reader.get_batch(list(range(start_frame, end_frame)))
                ref_frame = video_reader_refimg.get_batch(list(range(start_frame, end_frame)))
            else:
                indices = list(range(start_frame, self.max_num_frames))
                frames = video_reader.get_batch(indices)
                ref_frame = video_reader_refimg.get_batch(indices)

            # ref image for motion adapter
            # random frame을 선택 
            ref_idx = np.random.randint(0, len(ref_frame))
            mid_frame = copy(ref_frame[ref_idx])
            mid_frame = mid_frame.float()
            vit_frame = train_transforms(mid_frame)

            # Ensure that we don't go over the limit
            frames = frames[: self.max_num_frames]
            selected_num_frames = frames.shape[0]

            # Choose first (4k + 1) frames as this is how many is required by the VAE
            remainder = (3 + (selected_num_frames % 4)) % 4
            if remainder != 0:
                frames = frames[:-remainder]
            selected_num_frames = frames.shape[0]

            assert (selected_num_frames - 1) % 4 == 0
            
            # Training transforms
            frames = frames.float()
            frames = torch.stack([train_transforms(frame) for frame in frames], dim=0)
            videos.append(frames.permute(0, 3, 1, 2).contiguous())  # [F, C, H, W]
            vit_frames.append(vit_frame.permute(2, 0, 1).contiguous()) # [C, F, H]

        return videos, vit_frames


def log_validation(
    pipe,
    args,
    accelerator,
    pipeline_args,
    epoch,
    is_final_validation: bool = False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )
    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipe.scheduler.config:
        variance_type = pipe.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, **scheduler_args)
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    videos = []
    for _ in range(args.num_validation_videos):
        pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[0]
        pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_images)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)

        videos.append(image_pil)

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            video_filenames = []
            for i, video in enumerate(videos):
                prompt = (
                    pipeline_args["prompt"][:25]
                    .replace(" ", "_")
                    .replace(" ", "_")
                    .replace("'", "_")
                    .replace('"', "_")
                    .replace("/", "_")
                )
                filename = os.path.join(args.output_dir, f"{phase_name}_video_{i}_{prompt}.mp4")
                export_to_video(video, filename, fps=8)
                video_filenames.append(filename)

            tracker.log(
                {
                    phase_name: [
                        wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                        for i, filename in enumerate(video_filenames)
                    ]
                }
            )

    del pipe
    free_memory()

    return videos

def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds

def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
        tokenizer,
        text_encoder,
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        max_sequence_length=max_sequence_length,
        device=device,
        dtype=dtype,
        text_input_ids=text_input_ids,
    )
    return prompt_embeds

def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    if requires_grad:
        prompt_embeds = encode_prompt(
            tokenizer,
            text_encoder,
            prompt,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
    else:
        with torch.no_grad():
            prompt_embeds = encode_prompt(
                tokenizer,
                text_encoder,
                prompt,
                num_videos_per_prompt=1,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
    return prompt_embeds

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and args.optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


def main(args):

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

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
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    ## LoRA 외에 모델 동결
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(False)
    vae.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype),

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    ## 모델에 LoRA 등록 #########################################################################
    transformer_lora_config_id = LoraConfig(
        r=args.id_rank,
        lora_alpha=args.id_lora_alpha,
        init_lora_weights=True,
        target_modules=["to_k", "to_q", "ff.net.0.proj"],
    )
    transformer.add_adapter(transformer_lora_config_id, adapter_name="id_adapter")
    
    ## Identity adapter #########################################################################
    transformer_lora_config_motion = LoraConfig(
        r=args.motion_rank,
        lora_alpha=args.motion_lora_alpha,
        init_lora_weights=True,
        target_modules=["to_v", "to_out.0", "ff.net.0.proj", "ff.net.2.proj"],
    )
    transformer.add_adapter(transformer_lora_config_motion, adapter_name="motion_adapter") 

    # transformer_lora_config_shared = LoraConfig(
    #     r=args.rank,
    #     # args.rank,
    #     lora_alpha=args.lora_alpha,
    #     init_lora_weights=True,
    #     target_modules=["ff.net.0.proj"],
    # )
    # transformer.add_adapter(transformer_lora_config_shared, adapter_name="shared_adapter") 

    # 초기화 
    for name, param in transformer.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = False
        else:
            # Base model(DiT) 가중치는 당연히 고정
            param.requires_grad = False

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # accelerator.save_state(...)가 호출될 때 실행
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    unwrapped_model = unwrap_model(model)
                    active_adapters = unwrapped_model.peft_config.keys()

                    if "id_adapter" in active_adapters:
                        transformer_lora_layers_id = get_peft_model_state_dict(unwrapped_model, adapter_name="id_adapter")
                    if "motion_adapter" in active_adapters:
                        transformer_lora_layers_motion = get_peft_model_state_dict(unwrapped_model, adapter_name="motion_adapter")
                    if "shared_adapter" in active_adapters:
                        transformer_lora_layers_shared = get_peft_model_state_dict(unwrapped_model, adapter_name="shared_adapter")
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            if "id_adapter" in active_adapters:
                CogVideoXPipeline.save_lora_weights(
                    os.path.join(output_dir, "id_adapter"),
                    transformer_lora_layers=transformer_lora_layers_id,
                )
            if "motion_adapter" in active_adapters:
                CogVideoXPipeline.save_lora_weights(
                    os.path.join(output_dir, "motion_adapter"),
                    transformer_lora_layers=transformer_lora_layers_motion,
                )
            if "shared_adapter" in active_adapters:
                CogVideoXPipeline.save_lora_weights(
                    os.path.join(output_dir, "shared_adapter"),
                    transformer_lora_layers=transformer_lora_layers_shared,
                )
    # accelerator.load_state(...)가 호출될 때 실행
    def load_model_hook(models, input_dir):
        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"Unexpected save model: {model.__class__}")

        lora_state_dict = CogVideoXPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")

        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params([transformer_])

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            # train_batch_size 3
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    if args.mixed_precision == "fp16":
        # cast_training_params은 requires_grad=True인 파라미터만 캐스팅하지만,
        # 현재 모든 LoRA 파라미터가 requires_grad=False 상태이므로 직접 캐스팅
        for name, param in transformer.named_parameters():
            if "lora" in name.lower():
                param.data = param.data.to(torch.float32)

    ## p.requires_grad == True인 파라미터만 모으는 코드
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    ## Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    params_to_optimize = [transformer_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    # Dataset and DataLoader
    train_dataset_id = VideoDataset_for_id(
        instance_data_root=args.instance_data_root_id,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        caption_column=args.caption_column_id,
        video_column=args.video_column_id,
        height=args.height,
        width=args.width,
        fps=args.fps,
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
        height=args.height,
        width=args.width,
        fps=args.fps,
        max_num_frames=args.max_num_frames,
        skip_frames_start=args.skip_frames_start,
        skip_frames_end=args.skip_frames_end,
        cache_dir=args.cache_dir,
        id_token=args.id_token,
    )

    ## dataset vae encoding 

    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype).unsqueeze(0)
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = vae.encode(video).latent_dist
        return latent_dist

    train_dataset_id.instance_videos = [encode_video(video) for video in train_dataset_id.instance_videos]
    train_dataset_motion.instance_videos = [encode_video(video) for video in train_dataset_motion.instance_videos]


    def collate_fn(examples):
        videos = [example["instance_video"].sample() * vae.config.scaling_factor for example in examples]
        prompts = [example["instance_prompt"] for example in examples]

        videos = torch.cat(videos)
        videos = videos.permute(0, 2, 1, 3, 4)
        videos = videos.to(memory_format=torch.contiguous_format).float()

        return {
            "videos": videos,
            "prompts": prompts,
        }

    # dataset이 sample 단위 -> dataloader로 묶을 때 collate_fn 기준으로 batch 묶음
    train_dataloader_id = DataLoader(
        train_dataset_id,
        batch_size=args.train_batch_size_other, # .sh defualt 1, 매 샘플 하나만 보고 학습
        shuffle=True,  
        collate_fn=collate_fn, 
        num_workers=args.dataloader_num_workers,
    )

    train_dataloader_motion = DataLoader(
        train_dataset_motion,
        batch_size=args.train_batch_size,
        shuffle=True,  
        collate_fn=collate_fn, 
        num_workers=args.dataloader_num_workers,
    )

    length_dataloader = len(train_dataloader_id)
    length_dataset = len(train_dataset_id)+len(train_dataset_motion)

    # Scheduler and math around the number of training steps.
    # 한 epoch에 필요한 iteration 수
    num_update_steps_per_epoch = math.ceil(length_dataloader / args.gradient_accumulation_steps)

    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch

    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
            name=args.lr_scheduler,
            optimizer=optimizer,
            total_num_steps=args.max_train_steps * accelerator.num_processes,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    transformer, optimizer, train_dataloader_id, train_dataloader_motion, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader_id, train_dataloader_motion, lr_scheduler
    )

    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox-lora"
        accelerator.init_trackers(tracker_name, config=vars(args))


    ## Train!
    # total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # total_batch_size: 한번의 optimizer step을 학습하기 위해 모델이 실제로 본 총 데이터 개수
    # train_batch_size: gpu가 한번에 처리하는 batch 크기, num_processes: 사용중인 gpu 개수, gradient_accumulation_steps: 여러개의 미니 매치를 쌓아서 1step으로 만드는 단계 
    # train_batch_size 3, num_processes 1, gradient_accumulation_steps 1

    ## 학습 가능한 파라미터 수
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {length_dataset}") # 전체 데이터 수 
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}") # gpu 1대가 즉시 처리하는 batch size
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}") 
    logger.info(f"  Total optimization steps = {args.max_train_steps}") # 1000
    global_step = 0 # 현재까지 수행한 optimizer step 수
    # first_epoch = 0 # dataset 반복 횟수
    # initial_global_step은 현재 iteration 수 

    def next_or_restart(iterator, dataloader):
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(dataloader)
            return next(iterator), iterator
    
    # Potentially load in the weights and states from a previous save
    if not args.resume_from_checkpoint:
        initial_global_step = 0
    else:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")

            ## load_model_hook 호출되서 실행됨 
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config
   
    # prompt embeddings 미리 캐싱 (매 step마다 text encoder forward pass 방지)
    prompt_embeds_cache = {}
    all_prompts = (
        [p for batch in train_dataloader_id for p in batch["prompts"]]
        + [p for batch in train_dataloader_motion for p in batch["prompts"]]
    )
    for prompt in set(all_prompts):
        prompt_embeds_cache[prompt] = compute_prompt_embeddings(
            tokenizer,
            text_encoder,
            [prompt],
            model_config.max_text_seq_length,
            accelerator.device,
            weight_dtype,
            requires_grad=False,
        )

    # rotary positional embeddings 캐싱 (height/width/frames 고정이므로 매 step 재계산 불필요)
    # num_frames는 VAE 인코딩 후의 latent frame 수 (temporal compression 반영)
    latent_num_frames = train_dataset_id.instance_videos[0].sample().shape[1]
    cached_rotary_emb = (
        prepare_rotary_positional_embeddings(
            height=args.height,
            width=args.width,
            num_frames=latent_num_frames,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            patch_size=model_config.patch_size,
            attention_head_dim=model_config.attention_head_dim,
            device=accelerator.device,
        )
        if model_config.use_rotary_positional_embeddings
        else None
    )

    # max step trainig
    iter_id = iter(train_dataloader_id)
    iter_motion = iter(train_dataloader_motion)

    ## 
    while initial_global_step < args.max_train_steps:
        
        transformer.train()

        ## DataLoader에서 배치 가져오기
        ## id의 경우 1개, motion의 경우 3개 샘플을 가져옴
        batch_id, iter_id = next_or_restart(iter_id, train_dataloader_id)
        batch_motion, iter_motion = next_or_restart(iter_motion, train_dataloader_motion)
        
        ## 각 step에서 id 로더와 motion 로더 중 하나를 선택
        use_motion = random.random() < args.motion_ratio
        batch = batch_motion if use_motion else batch_id
        
        ## adapter_flag에 따라 적절한 adapter 활성화 및 학습 가능 설정
        if use_motion:
            transformer.set_adapters(["motion_adapter", "shared_adapter"])
            for name, param in transformer.named_parameters():
                if "motion_adapter" in name or "shared_adapter" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            transformer.set_adapters(["id_adapter", "shared_adapter"])
            for name, param in transformer.named_parameters():
                if "id_adapter" in name or "shared_adapter" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        models_to_accumulate = [transformer]

        ## gradient 누적 
        with accelerator.accumulate(models_to_accumulate):

            model_input = batch["videos"].to(dtype=weight_dtype)  # [B, F, C, H, W]
            prompts = batch["prompts"]

            ## encode prompts (캐시 사용)
            prompt_embeds = torch.cat([prompt_embeds_cache[p] for p in prompts], dim=0)

            ## noisy latent 생성
            noise = torch.randn_like(model_input)
            batch_size, num_frames, num_channels, height, width = model_input.shape

            ## 타임스텝 무작위로 생성
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (batch_size,), device=model_input.device
            )
            timesteps = timesteps.long()

            ## rotary embeds (캐시 사용)
            image_rotary_emb = cached_rotary_emb

            ## timestep에 따라 노이즈 추가 
            # model_input: clean VAE latent
            # noise: noise
            # noisy_model_input: noisy latent
            noisy_model_input = scheduler.add_noise(model_input, noise, timesteps)

            ## output 예측 (noise_pred)
            model_output = transformer(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
                # y_image=y_image,
            )[0]


            # datatypes = batch["datatype"] 
            # video_indices = [i for i, t in enumerate(datatypes) if t == 'video']
            # AiT Loss
            # if use_motion:
            #    B = model_output.shape[0]
             #   offset_noise = torch.randn(B, 1, 1, height, width, device=model_input.device, dtype=model_output.dtype)
              #  model_output = model_output + offset_noise

            ## velocity 예측
            ## diffuser default setting
            ## model_output: sample
            ## noisy_model_input: noise
            model_pred = scheduler.get_velocity(model_output, noisy_model_input, timesteps)

            alphas_cumprod = scheduler.alphas_cumprod[timesteps]
            weights = 1 / (1 - alphas_cumprod)
            while len(weights.shape) < len(model_pred.shape):
                weights = weights.unsqueeze(-1)

            target = model_input

            ## loss 계산
            ## “모델이 예측한 (velocity-form으로 변환된) denoised 동영상 model_pred가,
            ## 실제 clean VAE latent model_input와 최대한 같아지도록” 
            ## x₀-loss
            loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
            loss = loss.mean()
            accelerator.backward(loss)
            

            ###############################################
            # ★★★ LoRA 학습 여부 확인 코드 넣는 위치 ★★★
            ###############################################
            # if accelerator.sync_gradients:
            #     print("\n=== LoRA Grad Check ===")
            #     for name, p in transformer.named_parameters():
            #         if "lora" in name and p.requires_grad:
            #             if p.grad is None:
            #                 print(f"❌ NO GRAD: {name}")
            #             else:
            #                 print(f"✔ grad OK: {name}, grad_mean={p.grad.abs().mean().item():.6f}")
            ###############################################


            ## === 여기서 optimizer param_group을 재구성 ===
            if accelerator.sync_gradients and accelerator.state.deepspeed_plugin is None:
                trainable_params = [p for p in transformer.parameters() if p.requires_grad]
                optimizer.param_groups = []
                optimizer.add_param_group({"params": trainable_params, "lr": args.learning_rate})

                params_to_clip = trainable_params
                accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()

                lr_scheduler.step()

            elif accelerator.sync_gradients and accelerator.state.deepspeed_plugin is not None:
                # DeepSpeed인 경우 optimizer/step은 deepspeed가 관리, 여기서는 scheduler만 스텝
                lr_scheduler.step()

        ## gradient 누적하는 마지막 step인지 확인 
        if accelerator.sync_gradients:
            progress_bar.update(1)

            ## optimizer.step() 수행 후 1 step 증가
            global_step += 1
            
            if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:

                # 지정한 step마다 checkpoint 저장
                if global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            removing_checkpoints = checkpoints[0:num_to_remove]

                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)
                    # 새로운 checkpoint 저장 경로 생성 
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path) # -> save_model_hook 호출되서 실행됨 
                    logger.info(f"Saved state to {save_path}")

        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)

        ## Validation용 Inference 코드
        ## 현재 initial_global_step이 inference_steps의 배수 일때만 validation용 inference 수행
        if accelerator.is_main_process:
            if args.inf_eva_prompts and initial_global_step % args.inference_steps == 0:
                # 검증용 새 파이프라인 생성 
                pipe = CogVideoXPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    text_encoder=unwrap_model(text_encoder),
                    scheduler=scheduler,
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)
                for validation_prompt in validation_prompts:
                    pipeline_args = {
                        "prompt": validation_prompt,
                        "guidance_scale": args.guidance_scale,
                        "use_dynamic_cfg": args.use_dynamic_cfg,
                        "height": args.height,
                        "width": args.width,
                    }
                    ### inference step 주기마다 시행되는 training 중간의 validation
                    validation_outputs = log_validation(
                        pipe=pipe,
                        args=args,
                        accelerator=accelerator,
                        pipeline_args=pipeline_args,
                        epoch=initial_global_step, 
                        # 원래 epoch이였는데 initial_global_step이 맞는 것 같음. 
                    )
        
        ## 1 step 증가
        initial_global_step += 1
        print("\033[1;31m training step: \033[0m", initial_global_step)

    ## Save the lora layers
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        transformer = unwrap_model(transformer)
        dtype = (
            torch.float16
            if args.mixed_precision == "fp16"
            else torch.bfloat16
            if args.mixed_precision == "bf16"
            else torch.float32
        )
        transformer = transformer.to(dtype)

        active_adapters = transformer.peft_config.keys()

        # 학습 성공 시 train_args.json 저장
        import json
        args_dict = vars(args).copy()
        args_dict["train_script"] = "train/train_lora.py"
        args_dict["adapter_config"] = {
            "id_adapter":     {"rank": args.id_rank or args.rank, "lora_alpha": args.id_lora_alpha or args.lora_alpha, "target_modules": ["to_k", "to_q", "ff.net.0.proj"]},
            "motion_adapter": {"rank": args.motion_rank or args.rank, "lora_alpha": args.motion_lora_alpha or args.lora_alpha, "target_modules": ["to_v", "to_out.0", "ff.net.0.proj", "ff.net.2.proj"]},
            "shared_adapter": {"rank": args.shared_rank or args.rank, "lora_alpha": args.shared_lora_alpha or args.lora_alpha, "target_modules": ["ff.net.0.proj"]},
        }
        with open(os.path.join(args.output_dir, "train_args.json"), "w") as f:
            json.dump(args_dict, f, indent=4)

        if "id_adapter" in active_adapters:
            transformer_lora_layers_id = get_peft_model_state_dict(transformer, adapter_name="id_adapter")
            CogVideoXPipeline.save_lora_weights(
                save_directory=os.path.join(args.output_dir, "id_adapter"),
                transformer_lora_layers=transformer_lora_layers_id,
            )
        if "motion_adapter" in active_adapters:
            transformer_lora_layers_motion = get_peft_model_state_dict(transformer, adapter_name="motion_adapter")
            CogVideoXPipeline.save_lora_weights(
                save_directory=os.path.join(args.output_dir, "motion_adapter"),
                transformer_lora_layers=transformer_lora_layers_motion,
            )
        if "shared_adapter" in active_adapters:
            transformer_lora_layers_shared = get_peft_model_state_dict(transformer, adapter_name="shared_adapter")
            CogVideoXPipeline.save_lora_weights(
                save_directory=os.path.join(args.output_dir, "shared_adapter"),
                transformer_lora_layers=transformer_lora_layers_shared,
            )

        # Cleanup trained models to save memory
        del transformer
        free_memory()

        # Final test inference
        pipe = CogVideoXPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=weight_dtype,
        )
        
        pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

        if args.enable_slicing:
            pipe.vae.enable_slicing()
        if args.enable_tiling:
            pipe.vae.enable_tiling()

        # Load LoRA weights
        lora_scaling = args.lora_alpha / args.rank
        pipe.load_lora_weights(args.output_dir, adapter_name="cogvideox-lora")
        pipe.set_adapters(["cogvideox-lora"], [lora_scaling])

        ## 
        validation_outputs = []

        if args.validation_prompt and args.num_validation_videos > 0:
            validation_prompts = args.validation_prompt.split(args.validation_prompt_separator)

            for validation_prompt in validation_prompts:
                pipeline_args = {
                    "prompt": validation_prompt,
                    "guidance_scale": args.guidance_scale,
                    "use_dynamic_cfg": args.use_dynamic_cfg,
                    "height": args.height,
                    "width": args.width,
                }

                # log_validation 함수 내부에서 video 저장됨 
                video = log_validation(
                    pipe=pipe,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=initial_global_step, # epoch에서 내가 수정함. 
                    is_final_validation=True,
                )
                validation_outputs.extend(video)
                # validation output 저장은 되는데, 출력은 안됨.

    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    try:
        main(args)
    except Exception as e:
        if args.output_dir and os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
            print(f"학습 실패로 출력 폴더 삭제: {args.output_dir}")
        raise
