from __future__ import annotations

import gc
from typing import Sequence

import numpy as np
import torch

from lerobot.rewards.topreward.configuration_topreward import TOPRewardConfig
from lerobot.rewards.topreward.modeling_topreward import TOPRewardModel
from lerobot.rewards.topreward.processor_topreward import (
    TOPRewardEncoderProcessorStep,
)


_reward_model: TOPRewardModel | None = None
_encoder: TOPRewardEncoderProcessorStep | None = None
_config: TOPRewardConfig | None = None
_loaded_model_path: str | None = None


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def unload_model() -> None:
    global _reward_model, _encoder, _config, _loaded_model_path

    if _reward_model is not None:
        try:
            _reward_model.to("cpu")
        except Exception:
            pass

    _reward_model = None
    _encoder = None
    _config = None
    _loaded_model_path = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_model(
    model_path: str,
    *,
    verbose: bool = False,
) -> tuple[
    TOPRewardModel,
    TOPRewardEncoderProcessorStep,
    TOPRewardConfig,
]:
    global _reward_model, _encoder, _config, _loaded_model_path

    if (
        _reward_model is not None
        and _encoder is not None
        and _config is not None
        and _loaded_model_path == model_path
    ):
        return _reward_model, _encoder, _config

    if _reward_model is not None:
        unload_model()

    device = _default_device()

    if verbose:
        print(f"Loading LeRobot TOPReward from {model_path} on {device}...")

    # max_frames=None prevents the LeRobot processor from silently truncating
    # prefixes after RewardGen has already selected its sparse frame anchors.
    config = TOPRewardConfig(
        vlm_name=model_path,
        device=device,
        max_frames=None,
    )

    model = TOPRewardModel(config).to(device).eval()

    encoder = TOPRewardEncoderProcessorStep(
        vlm_name=config.vlm_name,
        image_key=config.image_key,
        task_key=config.task_key,
        default_task=config.default_task,
        max_frames=None,
        fps=config.fps,
        prompt_prefix=config.prompt_prefix,
        prompt_suffix_template=config.prompt_suffix_template,
        add_chat_template=config.add_chat_template,
        max_length=config.max_input_length,
    )

    _reward_model = model
    _encoder = encoder
    _config = config
    _loaded_model_path = model_path

    return model, encoder, config


def _as_uint8_rgb_video(frames: Sequence[np.ndarray]) -> torch.Tensor:
    video = np.asarray(frames)

    if video.ndim != 4 or video.shape[-1] not in (1, 3):
        raise ValueError(
            "TOPReward expects frames shaped (T, H, W, C); "
            f"received {video.shape}"
        )

    if np.issubdtype(video.dtype, np.floating):
        if video.size and video.max() <= 1.0:
            video = video * 255.0

    video = np.clip(video, 0, 255).astype(np.uint8)

    # LeRobot's TOPReward processor accepts (B, T, C, H, W).
    return torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()


def topreward(
    frames: Sequence[np.ndarray],
    instruction: str,
    verbose: bool = False,
    model_path: str | None = None,
) -> list[float]:
    """
    Evaluate every prefix of RewardGen's sampled frames.

    The returned list has exactly len(frames) elements. RewardGen core then
    interpolates these sparse values onto every original video timestep.
    """
    from rewardgen.utils.model_utils import get_model_dir

    if len(frames) == 0:
        raise ValueError("TOPReward received an empty video")

    if model_path is None:
        model_path = get_model_dir("topreward")

    model, encoder, config = load_model(
        model_path,
        verbose=verbose,
    )

    video = _as_uint8_rgb_video(frames)
    raw_log_probs: list[float] = []

    for prefix_length in range(1, len(video) + 1):
        prefix = video[:prefix_length].unsqueeze(0)

        # String keys are intentional: TransitionKey is a str Enum, so this
        # works across LeRobot versions without importing a relocated type.
        transition = {
            "observation": {
                config.image_key: prefix,
            },
            "complementary_data": {
                config.task_key: instruction,
            },
        }

        encoded_transition = encoder(transition)
        batch = encoded_transition["observation"]

        with torch.inference_mode():
            reward = model.compute_reward(batch)

        value = float(reward.detach().cpu().reshape(-1)[0])
        raw_log_probs.append(value)

        if verbose:
            print(
                f"TOPReward prefix {prefix_length}/{len(video)}: "
                f"log P(True) = {value:.6f}"
            )

    raw = np.asarray(raw_log_probs, dtype=np.float64)

    if not np.all(np.isfinite(raw)):
        raise RuntimeError(
            f"TOPReward produced non-finite values: {raw.tolist()}"
        )

    value_range = float(raw.max() - raw.min())

    # This follows the behavior in your example. A completely flat signal
    # becomes 100% everywhere instead of becoming all zeros.
    if value_range <= 1e-12:
        normalized = np.ones_like(raw)
    else:
        normalized = (raw - raw.min()) / value_range

    return (normalized * 100.0).tolist()

