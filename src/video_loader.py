# src/video_loader.py
"""Video loading and frame extraction utilities."""

import os
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np
from PIL import Image


class VideoLoader:
    """Load and sample frames from video files."""

    def __init__(
        self,
        video_dir: str,
        frames_per_video: int = 8,
        sampling_strategy: str = "uniform",
        target_fps: Optional[float] = None,
    ):
        """Initialize video loader.

        Args:
            video_dir: Directory containing video files.
            frames_per_video: Number of frames to sample per video.
            sampling_strategy: Frame sampling strategy (uniform, keyframe, fps_based).
            target_fps: Target FPS for fps_based sampling.
        """
        self.video_dir = Path(video_dir)
        self.frames_per_video = frames_per_video
        self.sampling_strategy = sampling_strategy
        self.target_fps = target_fps or 1.0

        self.video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        self._video_files: list[Path] = []
        self._scan_videos()

    def _scan_videos(self) -> None:
        """Scan directory for video files."""
        self._video_files = sorted(
            [
                f
                for f in self.video_dir.iterdir()
                if f.suffix.lower() in self.video_extensions
            ]
        )

    @property
    def num_videos(self) -> int:
        """Return number of videos found."""
        return len(self._video_files)

    @property
    def video_files(self) -> list[Path]:
        """Return list of video file paths."""
        return self._video_files

    def _sample_frame_indices(self, total_frames: int, video_fps: float) -> list[int]:
        """Determine which frame indices to sample.

        Args:
            total_frames: Total number of frames in video.
            video_fps: Video frame rate.

        Returns:
            List of frame indices to extract.
        """
        if self.sampling_strategy == "uniform":
            if total_frames <= self.frames_per_video:
                return list(range(total_frames))
            step = total_frames / self.frames_per_video
            return [int(i * step) for i in range(self.frames_per_video)]

        elif self.sampling_strategy == "fps_based":
            frame_interval = int(video_fps / self.target_fps)
            frame_interval = max(1, frame_interval)
            indices = list(range(0, total_frames, frame_interval))
            return indices[: self.frames_per_video]

        else:  # keyframe - fallback to uniform
            if total_frames <= self.frames_per_video:
                return list(range(total_frames))
            step = total_frames / self.frames_per_video
            return [int(i * step) for i in range(self.frames_per_video)]

    def extract_frames(self, video_path: Path) -> tuple[list[Image.Image], dict]:
        """Extract frames from a single video.

        Args:
            video_path: Path to video file.

        Returns:
            Tuple of (list of PIL Images, video metadata dict).
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        metadata = {
            "filename": video_path.name,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration_sec": duration,
        }

        frame_indices = self._sample_frame_indices(total_frames, fps)
        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)

        cap.release()
        metadata["sampled_frames"] = len(frames)

        return frames, metadata

    def iterate_videos(
        self, sample_size: Optional[int] = None
    ) -> Generator[tuple[Path, list[Image.Image], dict], None, None]:
        """Iterate over videos, yielding extracted frames.

        Args:
            sample_size: Optional limit on number of videos to process.

        Yields:
            Tuple of (video_path, frames, metadata).
        """
        videos = self._video_files[:sample_size] if sample_size else self._video_files

        for video_path in videos:
            try:
                frames, metadata = self.extract_frames(video_path)
                yield video_path, frames, metadata
            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                continue
