from __future__ import annotations


import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

from rewardgen.robometer.robometer.data.dataset_types import ProgressSample, Trajectory

# from rewardgen.robometer.robometer.evals.eval_server import compute_batch_outputs
# Skipping import of cpp extensions due to incompatible torch version 2.9.0+cu128 for torchao version 0.16.0             Please see https://github.com/pytorch/ao/issues/2919 for more info
# robometer/robometer/utils/setup_utils.py:7: UserWarning: WARNING: Unsloth should be imported before [transformers, peft] to ensure all optimizations are applied. Your code may run slower or encounter memory issues without these optimizations.

# Please restructure your imports with 'import unsloth' at the top of your file.
#   from unsloth import FastVisionModel
# 🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
# 🦥 Unsloth Zoo will now patch everything to make training faster!
# 2026-03-16 13:41:22 | INFO     | [Rank 0] rewardgen.robometer.robometer.evals.eval_server:<module>:62 | rewardgen.robometer.robometer.eval_server logger initialized at level DEBUG


import warnings
import contextlib
import io
warnings.filterwarnings("ignore")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    from rewardgen.robometer.robometer.evals.eval_server import compute_batch_outputs


# from rewardgen.robometer.robometer.evals.eval_viz_utils import create_combined_progress_success_plot, extract_frames
from rewardgen.robometer.robometer.utils.save import load_model_from_hf
from rewardgen.robometer.robometer.utils.setup_utils import setup_batch_collator



reward_model = None
tokenizer = None
processor = None
exp_config = None
device = None


def unload_model():
    import gc, torch

    for name in ["reward_model", "tokenizer", "processor", "exp_config"]:
        globals()[name] = None

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
# def unload_model():
#     global reward_model, tokenizer, processor, exp_config
#     import gc
#     import torch
#     try: 
#         del reward_model
#     except:
#         pass
#     try:
#         del tokenizer
#     except:
#         pass
#     try:
#         del processor
#     except:
#         pass
#     try:
#         del exp_config
#     except:
#         pass
#     reward_model = None
#     tokenizer = None
#     processor = None
#     exp_config = None
#     gc.collect()
#     torch.cuda.empty_cache()
#     torch.cuda.ipc_collect()
    
def load_model(model_path):
    global reward_model, tokenizer, processor, exp_config
    if reward_model is None:
        # unload_sole()   
        print("\nLoading Robometer model ...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 
        exp_config, tokenizer, processor, reward_model = load_model_from_hf(
            model_path=model_path,
            device=device,
        )
        # 
        reward_model.eval()

def silent_call(fn, *args, **kwargs):
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), \
         contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)

def compute_rewards_per_frame_local(
    model_path: str,
    video_frames: np.ndarray,
    task: str,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load RBM from HuggingFace and run inference; return per-frame progress and success arrays."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # exp_config, tokenizer, processor, reward_model = load_model_from_hf(
    #     model_path=model_path,
    #     device=device,
    # )

    # reward_model.eval()
    if verbose:
        load_model(model_path)
    else:
        silent_call(load_model, model_path)
    # load_model(model_path, verbose=verbose)   # ensures model is loaded once
    global reward_model, tokenizer, processor, exp_config
    
    batch_collator = setup_batch_collator(processor, tokenizer, exp_config, is_eval=True)
    


    T = int(video_frames.shape[0])
    traj = Trajectory(
        frames=video_frames,
        frames_shape=tuple(video_frames.shape),
        task=task,
        id="0",
        metadata={"subsequence_length": T},
        video_embeddings=None,
    )
    progress_sample = ProgressSample(trajectory=traj, sample_type="progress")
    batch = batch_collator([progress_sample])

    # progress_inputs = batch["progress_inputs"]
    # for key, value in progress_inputs.items():
    #     if hasattr(value, "to"):
    #         progress_inputs[key] = value.to(device)

    # loss_config = getattr(exp_config, "loss", None)
    # is_discrete = (
    #     getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
    #     if loss_config else False
    # )
    # num_bins = (
    #     getattr(loss_config, "progress_discrete_bins", None)
    #     or getattr(exp_config.model, "progress_discrete_bins", 10)
    # )
    # if verbose:
    #     print("\nRunning Robometer inference.")
    #     results = compute_batch_outputs(
    #         reward_model,
    #         tokenizer,
    #         progress_inputs,
    #         sample_type="progress",
    #         is_discrete_mode=is_discrete,
    #         num_bins=num_bins,
    #     )
    # else:
    #     results = silent_call(compute_batch_outputs, reward_model, tokenizer, progress_inputs, sample_type="progress", is_discrete_mode=is_discrete, num_bins=num_bins)

    # progress_pred = results.get("progress_pred", [])
    
    
    progress_inputs = batch["progress_inputs"]

    # Determine the dtype used by the underlying Qwen model.
    backbone = getattr(reward_model, "model", reward_model)

    try:
        model_dtype = next(
            parameter.dtype
            for parameter in backbone.parameters()
            if parameter.is_floating_point()
        )
    except StopIteration:
        model_dtype = torch.float32

    # Move inputs to the GPU. Preserve integer tensors such as input_ids and
    # image_grid_thw, but cast floating-point image inputs to the model dtype.
    for key, value in progress_inputs.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device)

            if value.is_floating_point():
                value = value.to(dtype=model_dtype)

            progress_inputs[key] = value

    loss_config = getattr(exp_config, "loss", None)

    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower() == "discrete"
        if loss_config
        else False
    )

    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    if verbose:
        print("\nRunning Robometer inference.")

    autocast_enabled = (
        device.type == "cuda"
        and model_dtype in (torch.float16, torch.bfloat16)
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=model_dtype if autocast_enabled else torch.float32,
            enabled=autocast_enabled,
        ):
            if verbose:
                results = compute_batch_outputs(
                    reward_model,
                    tokenizer,
                    progress_inputs,
                    sample_type="progress",
                    is_discrete_mode=is_discrete,
                    num_bins=num_bins,
                )
            else:
                results = silent_call(
                    compute_batch_outputs,
                    reward_model,
                    tokenizer,
                    progress_inputs,
                    sample_type="progress",
                    is_discrete_mode=is_discrete,
                    num_bins=num_bins,
                )

    progress_pred = results.get("progress_pred", [])

    progress_array = (
        np.array(progress_pred[0], dtype=np.float32)
        if progress_pred and len(progress_pred) > 0
        else np.array([], dtype=np.float32)
    )

    outputs_success = results.get("outputs_success", {})
    success_probs = outputs_success.get("success_probs", []) if outputs_success else []
    success_array = (
        np.array(success_probs[0], dtype=np.float32)
        if success_probs and len(success_probs) > 0
        else np.array([], dtype=np.float32)
    )

    return progress_array, success_array


def compute_rewards_per_frame_batch_local(
    model_path: str,
    video_frames_batch,
    task,
    device: Optional[torch.device] = None,
    verbose: bool = False,
):
    """
    Run Robometer progress inference for multiple trajectories in one
    model forward.

    Args:
        video_frames_batch:
            List[np.ndarray], each shaped (T, H, W, C).

        task:
            Either one task string shared by the batch, or a list of
            task strings with len == batch size.

    Returns:
        progress_arrays: list[np.ndarray]
        success_arrays: list[np.ndarray]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if verbose:
        load_model(model_path)
    else:
        silent_call(load_model, model_path)

    global reward_model, tokenizer, processor, exp_config

    # Robometer's current progress-head code eventually stacks the
    # per-example predictions, so keeping the same number of frames
    # in each batch is safest.
    frame_counts = [len(frames) for frames in video_frames_batch]
    if len(set(frame_counts)) != 1:
        raise ValueError(
            "All trajectories in a Robometer batch must have the same "
            f"number of frames; got {frame_counts}"
        )

    if isinstance(task, str):
        tasks = [task] * len(video_frames_batch)
    else:
        tasks = list(task)
        if len(tasks) != len(video_frames_batch):
            raise ValueError(
                "Number of task strings must equal number of trajectories"
            )

    progress_samples = []

    for idx, (video_frames, task_i) in enumerate(
        zip(video_frames_batch, tasks)
    ):
        video_frames = np.asarray(video_frames)
        T = int(video_frames.shape[0])

        traj = Trajectory(
            frames=video_frames,
            frames_shape=tuple(video_frames.shape),
            task=task_i,
            id=str(idx),
            metadata={"subsequence_length": T},
            video_embeddings=None,
        )

        progress_samples.append(
            ProgressSample(
                trajectory=traj,
                sample_type="progress",
            )
        )

    # THIS is the important difference:
    #
    # old:
    #   batch_collator([one_sample])
    #
    # new:
    #   batch_collator([sample_0, ..., sample_B])
    batch_collator = setup_batch_collator(
        processor,
        tokenizer,
        exp_config,
        is_eval=True,
    )

    batch = batch_collator(progress_samples)
    progress_inputs = batch["progress_inputs"]

    for key, value in progress_inputs.items():
        if hasattr(value, "to"):
            progress_inputs[key] = value.to(
                device,
                non_blocking=True,
            )

    loss_config = getattr(exp_config, "loss", None)

    is_discrete = (
        getattr(loss_config, "progress_loss_type", "l2").lower()
        == "discrete"
        if loss_config
        else False
    )

    num_bins = (
        getattr(loss_config, "progress_discrete_bins", None)
        or getattr(exp_config.model, "progress_discrete_bins", 10)
    )

    if verbose:
        print(
            f"\nRunning Robometer inference "
            f"(batch_size={len(progress_samples)})."
        )

    # forward_model() already uses no_grad(), but inference_mode()
    # is appropriate for this pure inference path as well.
    with torch.inference_mode():
        if verbose:
            results = compute_batch_outputs(
                reward_model,
                tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=is_discrete,
                num_bins=num_bins,
            )
        else:
            results = silent_call(
                compute_batch_outputs,
                reward_model,
                tokenizer,
                progress_inputs,
                sample_type="progress",
                is_discrete_mode=is_discrete,
                num_bins=num_bins,
            )

    progress_pred = results.get("progress_pred", [])

    progress_arrays = [
        np.asarray(pred, dtype=np.float32)
        for pred in progress_pred
    ]

    outputs_success = results.get("outputs_success", {})
    success_probs = (
        outputs_success.get("success_probs", [])
        if outputs_success
        else []
    )

    success_arrays = [
        np.asarray(pred, dtype=np.float32)
        for pred in success_probs
    ]

    # Preserve one output per input even if success wasn't returned.
    if not success_arrays:
        success_arrays = [
            np.array([], dtype=np.float32)
            for _ in progress_arrays
        ]

    return progress_arrays, success_arrays


def robometer_batch(
    frames_batch,
    task_description,
    model_path=None,
    verbose=False,
):
    from rewardgen.utils.model_utils import get_model_dir

    if model_path is None:
        model_path = get_model_dir("robometer")

    rewards_batch, success_probs_batch = (
        compute_rewards_per_frame_batch_local(
            model_path=model_path,
            video_frames_batch=[
                np.asarray(frames)
                for frames in frames_batch
            ],
            task=task_description,
            verbose=verbose,
        )
    )

    rewards_batch = [
        (rewards * 100).tolist()
        for rewards in rewards_batch
    ]

    success_probs_batch = [
        (probs * 100).tolist()
        for probs in success_probs_batch
    ]

    return rewards_batch, success_probs_batch


# def robometer(frames_final, task_description, model_path=None, verbose=False):
#     # def robometer(video_path, task_description):
#     # frames = load_frames_input(
#     #     # str(args.video),
#     #     # fps=float(args.fps),
#     #     # max_frames=int(args.max_frames),
#     #     video_path,
#     #     fps=1000,
#     #     max_frames=1000,
#     # )
#     # # 
#     # # frames[0].shape
#     # frame_height, frame_width = frames[0].shape[:2]
#     # # 
#     # frames_final=frames
#     # if 'test_videos' in video_path:
#     #     if frame_width == 2*frame_height:
#     #         # extract left half of each frame since the video has side-by-side views and we only want to use the external view
#     #         frames_final=[]
#     #         for i in range(len(frames)):
#     #             frames_final.append(frames[i][:, :frames[i].shape[1]//2, :])
    
#     # frames_final[0].shape
#     from rewardgen.utils.model_utils import get_model_dir
#     if model_path is None:
#         model_path = get_model_dir("robometer")
#     # 
#     rewards, success_probs = compute_rewards_per_frame_local(
#         # model_path=args.model_path,
#         # model_path='../model_checkpoints/Robometer-4B/',
#         model_path=model_path,
#         video_frames=np.array(frames_final),
#         # task=args.task,
#         task=task_description,
#         verbose=verbose,
#     )
    
#     rewards = rewards.tolist() if rewards is not None else None
#     success_probs = success_probs.tolist() if success_probs is not None else None
    
#     # multiply by 100 to convert to percentage
#     if success_probs is not None:
#         success_probs = [prob * 100 for prob in success_probs]
#         rewards = [reward * 100 for reward in rewards]
#     # 
#     return rewards, success_probs


def robometer(
    frames_final,
    task_description,
    model_path=None,
    verbose=False,
):
    rewards_batch, success_batch = robometer_batch(
        [frames_final],
        task_description,
        model_path=model_path,
        verbose=verbose,
    )

    return rewards_batch[0], success_batch[0]

