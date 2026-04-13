"""
JointTuner_data → test_data 형식으로 변환하는 스크립트
- subject: .jpg 이미지 → 5프레임 반복 .mp4 + images/*.png + prompts.txt + videos.txt
- motion: .mp4 심볼릭 링크 + prompts.txt + videos.txt
"""

import os
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
import imageio

JOINTTUNER_ROOT = Path("/home/sbjeon/workspace/dataset/JointTuner_data")
TEST_DATA_ROOT = Path("/home/sbjeon/workspace/sb/train/test_data")

MOTION_PROMPTS = {
    "bear_walking": "a bear is walking",
    "boat_sailing": "a boat is sailing",
    "bus_traveling": "a bus is traveling",
    "dog_walking": "a dog is walking",
    "mallard_flying": "a mallard is flying",
    "person_dancing": "a person is dancing",
    "person_lifting_barbell": "a person is lifting a barbell",
    "person_playing_cello": "a person is playing cello",
    "person_playing_flute": "a person is playing flute",
    "person_twirling": "a person is twirling",
    "person_walking": "a person is walking",
    "train_turning": "a train is turning",
}


def jpg_to_mp4(jpg_path: Path, mp4_path: Path, n_frames: int = 5, fps: int = 8):
    """단일 jpg 이미지를 n_frames 반복 mp4로 변환"""
    img = Image.open(jpg_path).convert("RGB")
    frames = [np.array(img)] * n_frames
    imageio.mimwrite(str(mp4_path), frames, fps=fps, codec="libx264",
                     output_params=["-pix_fmt", "yuv420p"])


def setup_identity(subject_name: str, jpg_paths: list[Path], out_dir: Path):
    """identity/{subject_name}/ 디렉토리 생성"""
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    videos_dir = out_dir / "videos"
    images_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)

    video_entries = []
    prompt_entries = []

    for jpg_path in sorted(jpg_paths):
        stem = jpg_path.stem  # "00", "01", ...

        # images/*.png 복사
        img = Image.open(jpg_path).convert("RGB")
        img.save(images_dir / f"{stem}.png")

        # videos/*.mp4 생성 (5프레임 반복)
        mp4_path = videos_dir / f"{stem}.mp4"
        jpg_to_mp4(jpg_path, mp4_path, n_frames=5)

        video_entries.append(f"videos/{stem}.mp4")
        prompt_entries.append(f"{subject_name} *")

    # videos.txt
    (out_dir / "videos.txt").write_text("\n".join(video_entries) + "\n")
    # prompts.txt
    (out_dir / "prompts.txt").write_text("\n".join(prompt_entries) + "\n")

    print(f"  [identity] {subject_name}: {len(jpg_paths)} images → mp4")


def setup_motion(motion_name: str, mp4_paths: list[Path], out_dir: Path):
    """motion/{motion_name}/ 디렉토리 생성"""
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    prompt = MOTION_PROMPTS.get(motion_name, motion_name.replace("_", " "))
    video_entries = []
    prompt_entries = []

    for mp4_path in sorted(mp4_paths):
        stem = mp4_path.stem
        link_path = videos_dir / mp4_path.name
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        link_path.symlink_to(mp4_path.resolve())

        video_entries.append(f"videos/{mp4_path.name}")
        prompt_entries.append(prompt)

    (out_dir / "videos.txt").write_text("\n".join(video_entries) + "\n")
    (out_dir / "prompts.txt").write_text("\n".join(prompt_entries) + "\n")

    print(f"  [motion] {motion_name}: {len(mp4_paths)} videos")


def main():
    # ── Identity (subject) ──────────────────────────────────────────────
    print("=== Setting up identity ===")
    subject_root = JOINTTUNER_ROOT / "subject"
    for subject_group in sorted(subject_root.iterdir()):
        if not subject_group.is_dir():
            continue
        for subject_dir in sorted(subject_group.iterdir()):
            if not subject_dir.is_dir():
                continue
            jpg_files = sorted(subject_dir.glob("*.jpg")) or sorted(subject_dir.glob("*.png"))
            if not jpg_files:
                continue
            out_dir = TEST_DATA_ROOT / "identity" / subject_dir.name
            setup_identity(subject_dir.name, jpg_files, out_dir)

    # ── Motion ─────────────────────────────────────────────────────────
    print("\n=== Setting up motion ===")
    motion_root = JOINTTUNER_ROOT / "motion_49f"
    for motion_dir in sorted(motion_root.iterdir()):
        if not motion_dir.is_dir():
            continue
        mp4_files = sorted(motion_dir.glob("*.mp4"))
        if not mp4_files:
            continue
        out_dir = TEST_DATA_ROOT / "motion" / motion_dir.name
        setup_motion(motion_dir.name, mp4_files, out_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
