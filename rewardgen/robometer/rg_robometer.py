from __future__ import annotations

import gc
from typing import Sequence

import numpy as np
import torch

from lerobot.rewards.robometer import RobometerConfig, RobometerRewardModel
from lerobot.rewards.robometer.modeling_robometer import (
    ROBOMETER_FEATURE_PREFIX,
    ROBOMETER_INPUT_KEYS,
    decode_progress_outputs,
)
from lerobot.rewards.robometer.processor_robometer import (
    RobometerEncoderProcessorStep,
)


_reward_model: RobometerRewardModel | None = None
_encoder: RobometerEncoderProcessorStep | None = None
_loaded_model_path: str | None = None


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def unload_model() -> None:
    global _reward_model, _encoder, _loaded_model_path

    if _reward_model is not None:
        try:
            _reward_model.to("cpu")
        except Exception:
            pass

    _reward_model = None
    _encoder = None
    _loaded_model_path = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model(
    model_path: str,
    *,
    verbose: bool = False,
) -> tuple[RobometerRewardModel, RobometerEncoderProcessorStep]:
    global _reward_model, _encoder, _loaded_model_path

    if (
        _reward_model is not None
        and _encoder is not None
        and _loaded_model_path == model_path
    ):
        return _reward_model, _encoder

    if _reward_model is not None:
        unload_model()

    device = _default_device()

    if verbose:
        print(f"Loading LeRobot Robometer from {model_path} on {device}...")

    # max_frames=None is important: RewardGen performs its own sampling and
    # needs a prediction for every frame that it sends to Robometer.
    config = RobometerConfig(
        pretrained_path=model_path,
        device=device,
        reward_output="progress",
        max_frames=None,
    )

    model = RobometerRewardModel.from_pretrained(
        config.pretrained_path,
        config=config,
    )
    model.to(device).eval()

    encoder = RobometerEncoderProcessorStep(
        base_model_id=config.base_model_id,
        use_multi_image=config.use_multi_image,
        use_per_frame_progress_token=config.use_per_frame_progress_token,
        max_frames=None,
    )

    _reward_model = model
    _encoder = encoder
    _loaded_model_path = model_path

    return model, encoder


def _as_uint8_rgb_video(frames: Sequence[np.ndarray]) -> np.ndarray:
    video = np.asarray(frames)

    if video.ndim != 4 or video.shape[-1] not in (1, 3):
        raise ValueError(
            "Robometer expects frames shaped (T, H, W, C); "
            f"received {video.shape}"
        )

    if np.issubdtype(video.dtype, np.floating):
        if video.size and video.max() <= 1.0:
            video = video * 255.0

    return np.clip(video, 0, 255).astype(np.uint8)


def robometer_batch(
    frames_batch: Sequence[Sequence[np.ndarray]],
    task_description: str | Sequence[str],
    model_path: str | None = None,
    verbose: bool = False,
) -> tuple[list[list[float]], list[list[float]]]:
    """
    Return Robometer progress and success probability at every input frame.

    Outputs use RewardGen's existing 0-100 percentage convention.
    Interpolation to the original video length is handled by core.generate().
    """
    from rewardgen.utils.model_utils import get_model_dir

    if not frames_batch:
        return [], []

    if model_path is None:
        model_path = get_model_dir("robometer")

    model, encoder = load_model(model_path, verbose=verbose)

    videos = [_as_uint8_rgb_video(frames) for frames in frames_batch]

    frame_counts = [len(video) for video in videos]
    if any(count == 0 for count in frame_counts):
        raise ValueError("Robometer received an empty video")

    # LeRobot stacks per-sample dense outputs, so trajectories in the same
    # forward pass must have equal temporal lengths.
    if len(set(frame_counts)) != 1:
        raise ValueError(
            "All videos in one Robometer batch must have the same number "
            f"of frames; received {frame_counts}"
        )

    if isinstance(task_description, str):
        tasks = [task_description] * len(videos)
    else:
        tasks = list(task_description)
        if len(tasks) != len(videos):
            raise ValueError(
                "Number of task descriptions must match number of videos"
            )

    encoded = encoder.encode_samples(list(zip(videos, tasks, strict=True)))
    batch = {
        f"{ROBOMETER_FEATURE_PREFIX}{key}": value
        for key, value in encoded.items()
    }

    # Reconstruct the dense-input path used internally by LeRobot.
    inputs = {
        key: batch[f"{ROBOMETER_FEATURE_PREFIX}{key}"]
        for key in ROBOMETER_INPUT_KEYS
        if f"{ROBOMETER_FEATURE_PREFIX}{key}" in batch
    }

    device = next(model.model.parameters()).device
    inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        progress_logits, success_logits = model._compute_rbm_logits(inputs)

    decoded = decode_progress_outputs(
        progress_logits,
        success_logits,
        is_discrete_mode=model.config.use_discrete_progress,
    )

    progress = decoded["progress_pred"]
    success = decoded["success_probs"]

    if len(progress) != len(videos) or len(success) != len(videos):
        raise RuntimeError(
            "Robometer returned a different batch size than it received"
        )

    for index, expected_length in enumerate(frame_counts):
        if len(progress[index]) != expected_length:
            raise RuntimeError(
                f"Robometer returned {len(progress[index])} progress values "
                f"for video {index}, expected {expected_length}"
            )
        if len(success[index]) != expected_length:
            raise RuntimeError(
                f"Robometer returned {len(success[index])} success values "
                f"for video {index}, expected {expected_length}"
            )

    progress_percent = [
        (np.asarray(values, dtype=np.float32) * 100.0).tolist()
        for values in progress
    ]
    success_percent = [
        (np.asarray(values, dtype=np.float32) * 100.0).tolist()
        for values in success
    ]

    return progress_percent, success_percent


def robometer(
    frames_final: Sequence[np.ndarray],
    task_description: str,
    model_path: str | None = None,
    verbose: bool = False,
) -> tuple[list[float], list[float]]:
    progress, success = robometer_batch(
        [frames_final],
        task_description,
        model_path=model_path,
        verbose=verbose,
    )
    return progress[0], success[0]

