import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json

import gc
import torch

def load_video_frames(video_path):
    import cv2
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        frames.append(frame)
    cap.release()
    return frames


CURRENT_MODEL = None


def generate(
    model: str,
    task_description: str,
    video_paths: list = None,
    video_path: str = None,
    ###### optionl, in case you want to give two videos with different views instead of one video that has a single view or has combined side-by-side views
    video_view_external_paths: list = None, 
    video_view_wrist_paths: list = None,
    video_view_external_path: str = None, 
    video_view_wrist_path: str = None,
    ######
    ###### optionl, in case you want to give frames instead of video paths
    video_frames: list = None,
    video_frames_external: list = None,
    video_frames_wrist: list = None,
    ######
    view_type_per_video: list = None,
    view_type: str = None,
    num_reasoning_frames: int = 10,
    context_window: str = None,
    ###### for API-based models
    key: str = None,
    ######
    ###### optional, for local models
    model_path: str = None,
    ######
    verbose: bool = True,
    max_batch_size: int = 5
):
    # 
    global CURRENT_MODEL
    # 
    if CURRENT_MODEL != model and CURRENT_MODEL is not None:
        if verbose:
            print(f"Switching model from {CURRENT_MODEL} → {model}.")
            # if not 'gpt' in model and not 'gemini' in model:
            print(f"Unloading previous model {CURRENT_MODEL} from GPU to free up memory...\n")
        # 
        if CURRENT_MODEL.lower() == 'robometer':
            from rewardgen.robometer.rg_robometer import unload_model as unload_robometer
            unload_robometer()
        # 
        elif CURRENT_MODEL.lower() == 'sole-r1':
            from rewardgen.sole import unload_model as unload_sole
            unload_sole()
        # 
        elif CURRENT_MODEL.lower() == "roboreward":
            from rewardgen.roboreward import unload_model as unload_roboreward
            unload_roboreward()
        # 
        elif CURRENT_MODEL.lower() == "topreward":
            from rewardgen.topreward import unload_model as unload_topreward
            unload_topreward()
    elif CURRENT_MODEL is None:
        # if verbose:
        #     print("CURRENT_MODEL is None. Trying to unload all known models...\n")
        try:
            from rewardgen.robometer.rg_robometer import unload_model as unload_robometer
            unload_robometer()
        except Exception:
            pass
        try:
            from rewardgen.sole import unload_model as unload_sole
            unload_sole()
        except Exception:
            pass
        try:
            from rewardgen.roboreward import unload_model as unload_roboreward
            unload_roboreward()
        except Exception:
            pass
        try:
            from rewardgen.topreward import unload_model as unload_topreward
            unload_topreward()
        except Exception:
            pass
    # 
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    CURRENT_MODEL = model
    # 
    if 'gpt' in model.lower() or 'gemini' in model.lower():
        assert key is not None, 'API key must be provided for OpenAI and Google models'
    # 
    single_video = False
    if video_frames is not None:
        # if video_frames[0] is a list of frames, then we have multiple videos worth of frames. if video_frames is a single list of frames, then we have one video worth of frames.
        if isinstance(video_frames[0], list):
            single_video = False
        else:
            single_video = True
        # 
        if view_type_per_video is None and view_type is not None:
            if single_video:
                view_type_per_video = [view_type]
            else:
                view_type_per_video = [view_type] * len(video_frames)
        # 
    elif video_frames_external is not None or video_frames_wrist is not None:
        if video_frames_external is not None:
            if isinstance(video_frames_external[0], list):
                single_video = False
            else:
                single_video = True
        if video_frames_wrist is not None:
            if isinstance(video_frames_wrist[0], list):
                single_video = False
            else:
                single_video = True
        # 
        if view_type_per_video is None and view_type is not None:
            if single_video:
                view_type_per_video = [view_type]
            else:
                view_type_per_video = [view_type] * max([len(video_frames_external or []), len(video_frames_wrist or [])])
    else:
        if video_paths is None and video_path is not None:
            video_paths = [video_path]
            single_video = True
            # 
            if view_type_per_video is None and view_type is not None:
                view_type_per_video = [view_type] * len(video_paths)
        # 
        else:
            if video_view_external_paths is None and video_view_external_path is not None:
                video_view_external_paths = [video_view_external_path]
                single_video = True
            if video_view_wrist_paths is None and video_view_wrist_path is not None:
                video_view_wrist_paths = [video_view_wrist_path]
                single_video = True
            # 
            if view_type_per_video is None and view_type is not None:
                view_type_per_video = [view_type] * max([len(video_paths or []), len(video_view_external_paths or []), len(video_view_wrist_paths or [])])
    # 
    ############ EXTRACT VIDEO FRAMES FOR ALL VIDEOS AS A LIST OF LISTS    
    # 
    if video_frames is not None:
        if single_video:
            videos = [video_frames]
        else:
            videos = video_frames
    elif video_frames_external is not None or video_frames_wrist is not None:
        if video_frames_external is not None and video_frames_wrist is not None:
            if single_video:
                video_frames_external = [video_frames_external]
                video_frames_wrist = [video_frames_wrist]
            # 
            if len(video_frames_external) != len(video_frames_wrist):
                raise ValueError(f"Number of videos in video_frames_external {len(video_frames_external)} does not match number of videos in video_frames_wrist {len(video_frames_wrist)}")
            # 
            videos = []
            for video_idx in range(len(video_frames_external)):
                frames_external = video_frames_external[video_idx]
                frames_wrist = video_frames_wrist[video_idx]
                if not len(frames_external) == len(frames_wrist):
                    raise ValueError(f"Number of frames in external view video {len(frames_external)} does not match number of frames in wrist view video {len(frames_wrist)} for video_idx {video_idx}")
                frames_combined = [np.concatenate((frames_external[i], frames_wrist[i]), axis=1) for i in range(len(frames_external))]
                videos.append(frames_combined)
            view_type_per_video = ['external and wrist'] * len(videos)
        else:
            if video_frames_external is not None and video_frames_wrist is None:
                if single_video:
                    video_frames_external = [video_frames_external]
                videos = video_frames_external
                view_type_per_video = ['external'] * len(video_frames_external)
                if verbose:
                    print(f"Using videos from video_frames_external with view type 'external'")
            elif video_frames_external is None and video_frames_wrist is not None:
                if single_video:
                    video_frames_wrist = [video_frames_wrist]
                videos = video_frames_wrist
                view_type_per_video = ['wrist'] * len(video_frames_wrist)
                if verbose:
                    print(f"Using videos from video_frames_wrist with view type 'wrist'")
            else:
                raise ValueError("video_frames cannot be None if video_paths is None")
    else:
        if video_paths is None and video_view_external_paths is not None and video_view_wrist_paths is not None:
            # concatenate external and wrist view videos side by side and use that as input to the model 
            videos = []
            for video_idx in range(len(video_view_external_paths)):
                frames_external = load_video_frames(video_view_external_paths[video_idx])
                frames_wrist = load_video_frames(video_view_wrist_paths[video_idx])
                if not len(frames_external) == len(frames_wrist):
                    raise ValueError(f"Number of frames in external view video {len(frames_external)} does not match number of frames in wrist view video {len(frames_wrist)} for video_idx {video_idx}")
                frames_combined = [np.concatenate((frames_external[i], frames_wrist[i]), axis=1) for i in range(len(frames_external))]
                videos.append(frames_combined)
            view_type_per_video = ['external and wrist'] * len(videos)
        else:
            if video_paths is None and video_view_external_paths is not None and video_view_wrist_paths is None:
                video_paths = video_view_external_paths
                view_type_per_video = ['external'] * len(video_view_external_paths)
                if verbose:
                    print(f"Using videos from video_view_external_paths with view type 'external'")
            elif video_paths is None and video_view_external_paths is None and video_view_wrist_paths is not None:
                video_paths = video_view_wrist_paths
                view_type_per_video = ['wrist'] * len(video_view_wrist_paths)
                if verbose:
                    print(f"Using videos from video_view_wrist_paths with view type 'wrist'")
            elif video_paths is None:
                raise ValueError("video_paths cannot be None if video_view_external_paths and video_view_wrist_paths are both not provided")
            # 
            videos = []
            for video_path in video_paths:
                frames = load_video_frames(video_path)
                videos.append(frames)
    # 
    ############ DOWNSAMPLE TO NUM_REASONING_FRAMES (e.g. 10) REASONING FRAMES PER VIDEO
    downsampled_videos = []
    downsample_idx_list_list = []
    for video_idx in range(len(videos)):
        if num_reasoning_frames < len(videos[video_idx]):
            downsample_idx_list = np.linspace(0, len(videos[video_idx])-1, num=num_reasoning_frames, dtype=int)
        else:
            downsample_idx_list = np.arange(len(videos[video_idx]))
        downsampled_videos.append([videos[video_idx][i] for i in downsample_idx_list])
        downsample_idx_list_list.append(downsample_idx_list.tolist())
    # 
    # 
    if not model.lower() == "sole-r1":
        frame_height, frame_width = downsampled_videos[0][0].shape[:2]
        if video_path is not None and 'test_videos' in video_path:
            if frame_width == 2*frame_height:
                for video_idx in range(len(downsampled_videos)):
                    # extrect left half of each frame for 'external' (default) and right half for 'wrist' since the video has side-by-side views and we only want to use the external view
                    if view_type_per_video[video_idx] is None or view_type_per_video[video_idx] == 'external':  
                        frames_final=[]
                        for i in range(len(downsampled_videos[video_idx])):
                            frames_final.append(downsampled_videos[video_idx][i][:, :downsampled_videos[video_idx][i].shape[1]//2, :])
                        # 
                        downsampled_videos[video_idx] = frames_final
                    elif view_type_per_video[video_idx] == 'wrist':
                        frames_final=[]
                        for i in range(len(downsampled_videos[video_idx])):
                            frames_final.append(downsampled_videos[video_idx][i][:, downsampled_videos[video_idx][i].shape[1]//2:, :])
                        # 
                        downsampled_videos[video_idx] = frames_final
    ############ GENERATE REWARD AND REASONING TRACES FOR EACH VIDEO
    success_probs = None
    reasoning_traces = None
    rewards = None
    full_rewards = None
    full_reasoning_traces = None
    if 'gpt' in model.lower() or 'gemini' in model.lower():
        # 
        from rewardgen.api_models import api_models
        rewards = []
        reasoning_traces = []
        for video_idx in range(len(downsampled_videos)):
            if verbose: 
                print(f"Generating rewards for video {video_idx+1}/{len(downsampled_videos)} using {model}...")
            rewards_video_i, reasoning_traces_video_i = api_models(model.lower(), downsampled_videos[video_idx], task_description, key, verbose=verbose)
            rewards.append(rewards_video_i)
            reasoning_traces.append(reasoning_traces_video_i)
        # 
    elif model.lower() == 'sole-r1': 
        from rewardgen.sole import sole
        # from sole import load_model
        # load_model()
        # rewards, reasoning_traces = sole(downsampled_videos, task_description, view_type_per_video=view_type_per_video, context_window=['current', 'previous', 'first'], model_path=model_path)
        if len(downsampled_videos)>5:
            # 
            lst=downsampled_videos
            downsampled_videos_chunks_max_5 = [lst[i:i+max_batch_size] for i in range(0, len(lst), max_batch_size)]
            view_type_per_video_chunks_max_5 = [view_type_per_video[i:i+max_batch_size] for i in range(0, len(view_type_per_video), max_batch_size)]
        else:
            downsampled_videos_chunks_max_5 = [downsampled_videos]
            view_type_per_video_chunks_max_5 = [view_type_per_video]
        # 
        rewards = []
        reasoning_traces = []
        for chunk_idx in range(len(downsampled_videos_chunks_max_5)):
            if verbose: 
                print(f"Generating rewards for batch {chunk_idx+1}/{len(downsampled_videos_chunks_max_5)} of videos (max_batch_size = {max_batch_size}) using SOLE-R1")
            downsampled_videos_chunk = downsampled_videos_chunks_max_5[chunk_idx]
            view_type_per_video_chunk = view_type_per_video_chunks_max_5[chunk_idx]
            rewards_chunk, reasoning_traces_chunk = sole(downsampled_videos_chunk, task_description, view_type_per_video=view_type_per_video_chunk, context_window=['current', 'previous', 'first'], model_path=model_path, verbose=verbose)
            rewards += rewards_chunk
            reasoning_traces += reasoning_traces_chunk
        # 
    elif model.lower() == 'topreward':
        from rewardgen.topreward import topreward
        rewards = []
        for video_idx in range(len(downsampled_videos)):
            if verbose: 
                print(f"Generating rewards for video {video_idx+1}/{len(downsampled_videos)} using TOPReward...")
            rewards_video_i = topreward(downsampled_videos[video_idx], task_description, verbose=verbose, model_path=model_path)
            rewards.append(rewards_video_i)
    # 
    elif model.lower() == 'roboreward':
        from rewardgen.roboreward import roboreward, RoboRewardModel
        rewards = []
        for video_idx in range(len(downsampled_videos)):
            if verbose: 
                print(f"Generating rewards for video {video_idx+1}/{len(downsampled_videos)} using RoboReward...")
            if model_path is None:
                rewards_video_i = roboreward(downsampled_videos[video_idx], task_description, verbose=verbose)
            else:
                rewards_video_i = roboreward(downsampled_videos[video_idx], task_description, verbose=verbose, model=RoboRewardModel(model_name=model_path))
            rewards.append(rewards_video_i)
    # 
    elif model.lower() == 'robometer':
        from rewardgen.robometer.rg_robometer import robometer
        # 
        rewards = []
        success_probs = []
        # for video_path in video_paths:
        for video_idx in range(len(downsampled_videos)):
            if verbose: 
                print(f"Generating rewards for video {video_idx+1}/{len(downsampled_videos)} using Robometer...")
            # rewards_video_i, success_probs_video_i = robometer(video_path, task_description)
            rewards_video_i, success_probs_video_i = robometer(downsampled_videos[video_idx], task_description, model_path=model_path, verbose=verbose)
            rewards.append(rewards_video_i)
            success_probs.append(success_probs_video_i)
    # 
    else:
        raise ValueError(f'Unknown model: {model}')
    # 
    ############ INTERPOLATE REWARD AND REASONING TRACES TO ALL FRAMES IN THE VIDEO (e.g. via linear interpolation)
    full_rewards = []
    full_success_probs = []
    full_reasoning_traces = []
    for video_idx in range(len(videos)):
        if not len(downsample_idx_list_list[video_idx]) == len(rewards[video_idx]):
            assert False, f"Length of downsample_idx_list {len(downsample_idx_list_list[video_idx])} does not match length of valid_answer_list {len(rewards[video_idx])} for video_idx {video_idx}"
        # 
        if len(downsample_idx_list_list[video_idx]) != len(videos[video_idx]):
            target_indices = np.arange(len(videos[video_idx]))
            x = np.asarray(downsample_idx_list_list[video_idx], dtype=float)
            y = np.asarray(rewards[video_idx], dtype=float)
            interp_rewards = np.interp(target_indices, x, y)
            # [int(x) for x in interp_rewards.tolist()]
            # rewards[video_idx]
            full_rewards.append(interp_rewards.tolist())
            # 
            if success_probs is not None:
                y = np.asarray(success_probs[video_idx], dtype=float)
                interp_success_probs = np.interp(target_indices, x, y)
                full_success_probs.append(interp_success_probs.tolist())
            # 
            if reasoning_traces is not None:
                full_reasoning_traces_video_i = []
                for step_idx in target_indices:
                    # step_idx = 0
                    if step_idx in downsample_idx_list_list[video_idx]:
                        full_reasoning_traces_video_i.append(reasoning_traces[video_idx][downsample_idx_list_list[video_idx].index(step_idx)])
                    else:
                        full_reasoning_traces_video_i.append(" ")
                full_reasoning_traces.append(full_reasoning_traces_video_i)
            else:
                full_reasoning_traces = None
        else:
            full_rewards = rewards
            full_reasoning_traces = reasoning_traces
            full_success_probs = success_probs
    # 
    if full_reasoning_traces is not None:
        if not single_video:
            return full_rewards, full_reasoning_traces
        else:
            return full_rewards[0], full_reasoning_traces[0]
    elif success_probs is not None:
        if not single_video:
            return full_rewards, full_success_probs
        else:
            return full_rewards[0], full_success_probs[0]
    else:
        if not single_video:
            return full_rewards
        else:
            return full_rewards[0]


def extract_annotation(lerobot_dataset, model, annotation_version=None, annotation_subdir=None):
    # 
    if annotation_version is None:
        annotation_version = "v1"
    # 
    if annotation_subdir is None:
        annotation_subdir = lerobot_dataset.root / "with_reward"
    # 
    if not os.path.exists(annotation_subdir):
        raise ValueError(f"Expected annotation subdir {annotation_subdir} does not exist")
    # 
    import pandas as pd
    # parquet_path = os.path.join(lerobot_dataset.root, "data/chunk-000/file-000.parquet")
    parquet_path = os.path.join(annotation_subdir, "data/chunk-000/file-000.parquet")
    df = pd.read_parquet(parquet_path)
    episode_lengths = df.groupby("episode_index").size().values
    # 
    rewards_by_episode = []
    current_episode = []
    episode_id = 0
    step_counter = 0
    reward_column_name = f"rewards_{model.lower()}_{annotation_version}"
    rewards_column = df[reward_column_name].values
    # len(rewards_column)
    # 
    for i in range(len(rewards_column)):
        current_episode.append(rewards_column[i])
        step_counter += 1
        if step_counter == episode_lengths[episode_id]:
            rewards_by_episode.append(np.array(current_episode).tolist())
            current_episode = []
            step_counter = 0
            episode_id += 1
    # 
    return rewards_by_episode

def extract_frames(lerobot_dataset, observation_name=None):
    if observation_name is None:
        observation_name = set_observation_name(video_paths)
    # 
    paths = lerobot_dataset.get_episodes_file_paths()
    video_paths = [p for p in paths if p.endswith(".mp4")]
    video_paths = [os.path.join(lerobot_dataset.root, p) for p in video_paths]
    # 
    import pandas as pd
    parquet_path = os.path.join(lerobot_dataset.root, "data/chunk-000/file-000.parquet")
    df = pd.read_parquet(parquet_path)
    episode_lengths = df.groupby("episode_index").size().values
    # 
    import av
    frames_by_episode = []
    current_episode = []
    episode_id = 0
    frame_counter = 0
    for video_path_idx in range(len(video_paths)):
        if observation_name in video_paths[video_path_idx]:
            container = av.open(video_paths[video_path_idx])
            for frame in container.decode(video=0):
                img = frame.to_ndarray(format="rgb24")
                current_episode.append(img)
                frame_counter += 1
                if frame_counter == episode_lengths[episode_id]:
                    frames_by_episode.append(current_episode)
                    current_episode = []
                    frame_counter = 0
                    episode_id += 1
                    # break
    # 
    return frames_by_episode

def set_observation_name(video_paths):
    if any('observation.images.side' in x for x in video_paths):
        observation_name = 'observation.images.side'
    elif any('observation.images.top' in x for x in video_paths):
        observation_name = 'observation.images.top'
    else:
        observation_name = video_paths[0].split('videos/')[-1].split('/')[0]
    return observation_name


def annotate(
    model: str,
    task_description: str,
    lerobot_dataset,
    observation_name: str = None,
    annotation_version: str = None,
    num_reasoning_frames: int = 10,
    context_window: str = None,
    ###### for API-based models
    key: str = None,
    ######
    ###### optional, for local models
    model_path: str = None,
    annotation_subdir = None,
    verbose: bool = True,
):
    paths = lerobot_dataset.get_episodes_file_paths()
    video_paths = [p for p in paths if p.endswith(".mp4")]
    video_paths = [os.path.join(lerobot_dataset.root, p) for p in video_paths]
    # 
    if observation_name is None:
        observation_name = set_observation_name(video_paths)
    # 
    if 'wrist' in observation_name or 'hand' in observation_name:
        view_type = 'wrist'
    else:
        view_type = 'external'
    # 
    if annotation_version is None:
        annotation_version = "v1"
    # 
    if annotation_subdir is None:
        annotation_subdir = lerobot_dataset.root / "with_reward"
    # 
    import av
    frames_by_episode = extract_frames(lerobot_dataset=lerobot_dataset, observation_name=observation_name)
    # 
    if verbose:
        print(f"Extracted frames for {len(frames_by_episode)} episodes with observation name '{observation_name}' and view type '{view_type}'")
    # 
    rewards = None
    reasoning_traces = None
    success_probs = None
    if model.lower() in ["roboreward", "topreward"]:
        rewards = generate(
            model=model,  
            task_description=task_description, 
            video_frames=frames_by_episode, 
            view_type=view_type, 
            num_reasoning_frames=num_reasoning_frames,
        )
    elif model.lower() == "robometer":
        rewards, success_probs = generate(
            model=model,  
            task_description=task_description, 
            video_frames=frames_by_episode, 
            view_type=view_type, 
            num_reasoning_frames=num_reasoning_frames,
        )
    elif model.lower() == "sole-r1":
        rewards, reasoning_traces = generate(
            model=model,  
            task_description=task_description, 
            video_frames=frames_by_episode, 
            view_type=view_type, 
            num_reasoning_frames=num_reasoning_frames,
        )
    elif "gpt" in model.lower() or "gemini" in model.lower():
        rewards, reasoning_traces = generate(
            model=model,  
            task_description=task_description, 
            video_frames=frames_by_episode, 
            view_type=view_type, 
            num_reasoning_frames=num_reasoning_frames,
            key=key
        )
    # 
    # convert rewards to numpy array and flatten to 1D array of length num_frames
    # reward_values = np.concatenate(rewards).astype(np.float32)
    reward_values = np.concatenate(rewards).astype(np.float32).reshape(-1, 1)
    # reward_values = np.random.randn(num_frames, 1).astype(np.float32)
    # reward_values.shape
    # 
    reward_column_name = f"rewards_{model.lower()}_{annotation_version}"
    reward_feature_info = {
        "dtype": "float32",
        "shape": (1,),
        "names": None,
    }
    features = {
        reward_column_name: (reward_values, reward_feature_info),
    }
    # if not reasoning_traces is None:
    #     reasoning_traces_column_name = f"reasoning_traces_{model}_{annotation_version}"
    # if not success_probs is None:
    #     success_probs_column_name = f"success_probs_{model}_{annotation_version}"
    # 
    from lerobot.datasets.dataset_tools import add_features
    num_frames = lerobot_dataset.meta.total_frames
    # 
    if verbose:
        print(f"Adding reward annotations to dataset with {num_frames} frames using feature name '{reward_column_name}'")
        print(f"Output directory: {str(annotation_subdir)}")
    # 
    # if annotation_subdir already exists, read this as current dataset
    # if os.path.exists(annotation_subdir):
    #     from lerobot.datasets.lerobot_dataset import LeRobotDataset
    #     current_dataset = LeRobotDataset(annotation_subdir, video_backend="pyav")
    # else:
    #     current_dataset = lerobot_dataset
    # 
    # if os.path.exists('/data/sls/scratch/pschro/.cache/huggingface/lerobot/jackvial/so101_pickplace_recap_merged_v2/with_reward'):
    if os.path.exists(annotation_subdir):
        # mkdir annotation_subdir_tmp
        import shutil
        if os.path.exists(f"{annotation_subdir}_tmp"):
            shutil.rmtree(f"{annotation_subdir}_tmp")
        shutil.move(annotation_subdir, f"{annotation_subdir}_tmp")
        # 
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        current_dataset = LeRobotDataset(f"{annotation_subdir}_tmp", video_backend="pyav")
        # 
        import pandas as pd
        current_df = pd.read_parquet(os.path.join(current_dataset.root, "data/chunk-000/file-000.parquet"))
        # if reward_column_name already exists in current_df, remove it and save the modified parquet file back to disk
        if reward_column_name in current_df.columns:
            current_df = current_df.drop(columns=[reward_column_name])
            current_df.to_parquet(os.path.join(current_dataset.root, "data/chunk-000/file-000.parquet"), index=False)
        # 
        # f"{annotation_subdir}_tmp" -> annotation_subdir
        new_dataset = add_features(
            dataset=current_dataset,
            features=features,
            # output_dir=sample_dataset.root / "with_reward",
            output_dir = annotation_subdir
            # output_dir = f'{str(annotation_subdir)}_tmp'
        )
        # move the new dataset to the original with_reward directory
        # os.system(f"rm -rf {str(annotation_subdir)}")
        # os.system(f"mv {str(annotation_subdir)}_tmp {str(annotation_subdir)}")
        # change the new dataset root to the original with_reward directory
        new_dataset.root = str(annotation_subdir)
    else:
        current_dataset = lerobot_dataset
        # original dir -> annotation_subdir
        new_dataset = add_features(
            dataset=current_dataset,
            features=features,
            output_dir=annotation_subdir,
        )
    # new_dataset.root
    # 
    assert reward_values.shape[0] == num_frames, f"Number of reward values {reward_values.shape[0]} does not match number of frames {num_frames}"
    assert reward_column_name in new_dataset.meta.features
    assert new_dataset.meta.features[reward_column_name] == reward_feature_info
    assert len(new_dataset) == num_frames


# def shape_to_target(frame, target=384):
#     frame_height, frame_width = frame.shape[:2]
#     if frame_height > target:
#         frame = cv2.resize(frame, (frame_width*target//frame_height, target), interpolation=cv2.INTER_AREA)
#     if frame_height < target: 
#         frame = cv2.copyMakeBorder(frame, (target-frame_height)//2, (target-frame_height)//2, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
#     if frame_width < target:
#         frame = cv2.copyMakeBorder(frame, 0, 0, (target-frame_width)//2, (target-frame_width)//2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
#     return frame

# def shape_to_target(frame, target=768):
#     frame_height, frame_width = frame.shape[:2]
#     # Always scale so height = target
#     scale = target / frame_height
#     new_width = int(frame_width * scale)
#     frame = cv2.resize(frame, (new_width, target), interpolation=cv2.INTER_LINEAR)
#     # Optional: pad width if still smaller than target
#     if new_width < target:
#         pad = (target - new_width) // 2
#         frame = cv2.copyMakeBorder(
#             frame, 0, 0, pad, pad,
#             cv2.BORDER_CONSTANT, value=[255, 255, 255]
#         )
#     return frame



# - If height is below target_height, scale up while preserving aspect ratio,
#     but do not let width exceed max_width. If width hits max_width first,
#     pad top/bottom.
# - If width is below min_width, scale up while preserving aspect ratio,
#     but do not let height exceed target_height. If height hits target_height
#     first, pad left/right.
def resize_and_pad_to_target(
    frame,
    target_height=768,
    min_width=768,
    max_width=1536,
    pad_color=(255,255,255)
):
    if frame is None or frame.size == 0:
        raise ValueError("Input frame is empty.")
    h, w = frame.shape[:2]
    # Step 1: If height is too small, scale up toward target_height,
    # but do not allow width to exceed max_width.
    if h < target_height:
        scale_to_height = target_height / h
        scale_to_max_width = max_width / w
        scale = min(scale_to_height, scale_to_max_width)
        if scale > 1:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = frame.shape[:2]
    # Step 2: If width is too small, scale up toward min_width,
    # but do not allow height to exceed target_height.
    if w < min_width:
        scale_to_width = min_width / w
        scale_to_target_height = target_height / h
        scale = min(scale_to_width, scale_to_target_height)
        if scale > 1:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = frame.shape[:2]
    # Step 3: Clamp width if somehow rounding pushed it over max_width.
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(round(h * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
    # Step 4: Pad to target height and minimum width as needed.
    pad_top = max((target_height - h) // 2, 0)
    pad_bottom = max(target_height - h - pad_top, 0)
    pad_left = max((min_width - w) // 2, 0)
    pad_right = max(min_width - w - pad_left, 0)
    if pad_top or pad_bottom or pad_left or pad_right:
        frame = cv2.copyMakeBorder(
            frame,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=pad_color,
        )
    return frame



import copy


def video_plot(outputs, plot_save_path, video_path=None, video_view_external_path=None, video_view_wrist_path=None, video_frames=None, view_type=None, task_description=None, show_all_frames=False,
               fps_=2, wrap_width=26, font_scale=1, font_height=30, text_thickness=2, line_type=2, show_reasoning_traces=True, cfg=None, env_rew_lab='Ground-truth reward', 
               save_json = True, verbose=False):
    # 
    ############ EXTRACT VIDEO FRAMES FOR ALL VIDEOS AS A LIST OF LISTS    
    # 
    outputs = copy.deepcopy(outputs)
    for output in outputs:
        if output is None:
            # remove this output from outputs
            outputs.remove(output)
    # 
    json_save_path = plot_save_path.rsplit('.', 1)[0] + '.json'
    if save_json:
        if not os.path.exists(os.path.dirname(json_save_path)):
            os.makedirs(os.path.dirname(json_save_path))
        with open(json_save_path, 'w') as f:
            json.dump(outputs, f)
    # 
    if video_frames is None:
        if video_path is None and video_view_external_path is not None and video_view_wrist_path is not None:
            frames_external = load_video_frames(video_view_external_path[video_idx])
            frames_wrist = load_video_frames(video_view_wrist_path[video_idx])
            if not len(frames_external) == len(frames_wrist):
                raise ValueError(f"Number of frames in external view video {len(frames_external)} does not match number of frames in wrist view video {len(frames_wrist)} for video_idx {video_idx}")
            frame_list = [np.concatenate((frames_external[i], frames_wrist[i]), axis=1) for i in range(len(frames_external))]
            view_type = 'external and wrist'
        else:
            if video_path is None and video_view_external_path is not None and video_view_wrist_path is None:
                video_path = video_view_external_path
                view_type = 'external'
                if verbose:
                    print(f"Using videos from video_view_external_path with view type 'external'")
            elif video_path is None and video_view_external_path is None and video_view_wrist_path is not None:
                video_path = video_view_wrist_path
                view_type = 'wrist'
                if verbose:
                    print(f"Using videos from video_view_wrist_path with view type 'wrist'")
            elif video_path is None:
                raise ValueError("video_path cannot be None if video_view_external_path and video_view_wrist_path are both not provided (and video_frames is also None)")
            # 
            frame_list = load_video_frames(video_path)
    else:
        frame_list = video_frames
    #   
    # only show reasoning traces in video if only showing one model - if "reasoning_traces" in more than 2 outputs, set reasoning_traces to None since this is too much information to show in the video
    count_reasoning_traces = 0
    output_rewards_len = []
    output_rewards_concat = []
    final_reasoning_traces_idx_list = []
    for output in outputs:
        output_rewards_len.append(len(output["rewards"]))
        output_rewards_concat += output["rewards"]
        if "reasoning_traces" in output and output["reasoning_traces"] is not None:
            count_reasoning_traces += 1
            final_reasoning_traces = output["reasoning_traces"]
            final_reasoning_traces_idx_list = []
            for step_idx in range(len(final_reasoning_traces)):
                if step_idx==0 or not final_reasoning_traces[step_idx].strip() == "":
                    final_reasoning_traces_idx_list.append(step_idx)
    # 
    if not count_reasoning_traces == 1:
        show_reasoning_traces = False
    # 
    if show_reasoning_traces:
        if not show_all_frames:
            if verbose:
                print('Only showing reasoning traces for frames where reasoning traces are not empty. If you want to show reasoning traces for all frames, set show_all_frames to True.')
            output_rewards_len = []
            output_rewards_concat = []
            for output_idx in range(len(outputs)):
                if 'reasoning_traces' in outputs[output_idx] and outputs[output_idx]['reasoning_traces'] is not None:
                    try:
                        outputs[output_idx]['reasoning_traces'] = [outputs[output_idx]['reasoning_traces'][i] for i in final_reasoning_traces_idx_list]
                    except:
                        if verbose:
                            print(f"Error in indexing reasoning traces for output_idx {output_idx}. This may be due to a mismatch between the length of reasoning traces and the length of rewards. Setting reasoning traces to None for this output.")
                outputs[output_idx]['rewards'] = [outputs[output_idx]['rewards'][i] for i in final_reasoning_traces_idx_list]
                output_rewards_len.append(len(outputs[output_idx]['rewards']))
                output_rewards_concat += outputs[output_idx]['rewards']
            final_reasoning_traces = [final_reasoning_traces[i] for i in final_reasoning_traces_idx_list]
            frame_list = [frame_list[i] for i in final_reasoning_traces_idx_list]
    # 
    text_title=''
    if True:
        plt.rcParams.update({'font.size': 12})  # Adjust font size as needed
        #             # 
        frame = frame_list[0]
        # if isinstance(frame, Image.Image):
        #     frame = np.array(frame.convert("RGB"))
        #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #
        # target_height = frame.shape[0]
        # frame = shape_to_target(frame, target=target_height)
        target_height = 768
        frame = resize_and_pad_to_target(frame, target_height=target_height, min_width=target_height, max_width=target_height*2)
        frame_height, frame_width = frame.shape[:2]
        # 
        # if view_type in ['external and wrist']:
        #     if show_reasoning_traces:
        #         output_width = target_height*4
        #     else:
        #         output_width = target_height*3
        # else:
        #     if show_reasoning_traces:
        #         output_width = target_height*3
        #     else:
        #         output_width = target_height*2
        if show_reasoning_traces:
            output_width = frame_width + (target_height*2)
        else:
            # output_width = target_height*3
            output_width = frame_width + target_height
        # 
        if task_description is not None:
            top_pad = 50
            output_height = frame_height + top_pad
        else:
            output_height = frame_height
        # 
        print('output_width, output_height:', output_width, output_height)
        # plot_save_path = "/data/sls/scratch/pschro/rewardgen/test_videos/lerobot/jackvial_so101_pickplace_recap_merged_v2/observation_images_top_episode_{episode_idx}.mp4"
        if plot_save_path.endswith(".mp4"):
            plot_save_path = plot_save_path.replace(".mp4", ".avi")
        # 
        if not os.path.exists(os.path.dirname(plot_save_path)):
            os.makedirs(os.path.dirname(plot_save_path))
        # 
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(plot_save_path, fourcc, fps_, (frame_width, frame_height))
        out = cv2.VideoWriter(plot_save_path, fourcc, fps_, (output_width, output_height))
        # 
        fig, ax = plt.subplots(dpi=200)  # Increase DPI for higher resolution
        canvas = FigureCanvas(fig)
        # xdata, ydata, ydata2 = [], [], []
        xdata, ydata, ydata2, ydata3, ydata4, ydata5, ydata6, ydata7 = [], [], [], [], [], [], [], []
        # xlim_max = max(len(predicted_rewards), len(groundtruth_rewards))
        xlim_max = max(output_rewards_len)
        ax.set_xlim(0, xlim_max-1)
        ax.set_title(text_title)
        ymax=100
        # 
        # ax.set_ylim(min(predicted_rewards+groundtruth_rewards), 100)
        ax.set_ylim(min(output_rewards_concat), 100)
        ax.set_ylabel('Predicted task progress (%)')
        ax.set_xlabel('Step number')
        # 
        ln, = ax.plot([], [], 'r', linewidth=5,label=outputs[0]["model"])
        ln.set_color('darkblue')
        ln.set_markerfacecolor('darkblue')
        ln.set_markersize(5)
        # 
        ln2, ln3, ln4, ln5, ln6, ln7 = None, None, None, None, None, None
        if len(outputs) > 1:
            ln2, = ax.plot([], [], 'b', linewidth=5,label=outputs[1]["model"])
            ln2.set_color('darkred')
            ln2.set_markerfacecolor('darkred')
            ln2.set_markersize(5)
        if len(outputs) > 2:
            ln3, = ax.plot([], [], 'g', linewidth=5,label=outputs[2]["model"])
            ln3.set_color('darkgreen')
            ln3.set_markerfacecolor('darkgreen')
            ln3.set_markersize(5)
        if len(outputs) > 3:
            ln4, = ax.plot([], [], 'y', linewidth=5,label=outputs[3]["model"])
            ln4.set_color('orange')
            ln4.set_markerfacecolor('orange')
            ln4.set_markersize(5)
        if len(outputs) > 4:
            ln5, = ax.plot([], [], 'c', linewidth=5,label=outputs[4]["model"])
            ln5.set_color('cyan')
            ln5.set_markerfacecolor('cyan')
            ln5.set_markersize(5)
        if len(outputs) > 5:
            ln6, = ax.plot([], [], 'm', linewidth=5,label=outputs[5]["model"])
            ln6.set_color('magenta')
            ln6.set_markerfacecolor('magenta')
            ln6.set_markersize(5)
        if len(outputs) > 6:
            ln7, = ax.plot([], [], 'k', linewidth=5,label=outputs[6]["model"])
            ln7.set_color('black')
            ln7.set_markerfacecolor('black')
            ln7.set_markersize(5)
        if len(outputs) > 7:
            assert False, f"Plotting more than 7 outputs in the same video is not currently supported."
        # 
        if len(outputs) <= 3:
            ax.legend(fontsize='small')
        elif len(outputs) <= 4:
            ax.legend(fontsize='x-small')
        else:
            ax.legend(fontsize='xx-small')
        frame_number = 0
        data_point_number = 0
        while frame_number<xlim_max:
            frame = frame_list[frame_number]
            # if isinstance(frame, Image.Image):
            #     frame = np.array(frame.convert("RGB"))
            #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # 
            # frame = shape_to_target(frame, target=target_height)
            frame = resize_and_pad_to_target(frame, target_height=target_height, min_width=target_height, max_width=target_height*2)
            frame_height, frame_width = frame.shape[:2]
            fig.set_size_inches(target_height / fig.dpi, target_height / fig.dpi)  
            fig.tight_layout(pad=0.4)  # Adjust layout to ensure everything fits without being cut off
            # 
            if True:
                xdata.append(data_point_number)
                # if len(predicted_rewards) > data_point_number:
                if len(outputs[0]["rewards"]) > data_point_number:
                    ydata.append(outputs[0]["rewards"][data_point_number])
                    ln.set_data(xdata, ydata)
                    ax.draw_artist(ln)
                # 
                ax.draw_artist(ax.patch)
                # if len(groundtruth_rewards) > data_point_number:
                if ln2 is not None:
                    ydata2.append(outputs[1]["rewards"][data_point_number])
                    ln2.set_data(xdata, ydata2)
                    ax.draw_artist(ln2)
                if ln3 is not None:
                    ydata3.append(outputs[2]["rewards"][data_point_number])
                    ln3.set_data(xdata, ydata3)
                    ax.draw_artist(ln3)
                if ln4 is not None:
                    ydata4.append(outputs[3]["rewards"][data_point_number])
                    ln4.set_data(xdata, ydata4)
                    ax.draw_artist(ln4)
                if ln5 is not None:
                    ydata5.append(outputs[4]["rewards"][data_point_number])
                    ln5.set_data(xdata, ydata5)
                    ax.draw_artist(ln5)
                if ln6 is not None:
                    ydata6.append(outputs[5]["rewards"][data_point_number])
                    ln6.set_data(xdata, ydata6)
                    ax.draw_artist(ln6)
                if ln7 is not None:    
                    ydata7.append(outputs[6]["rewards"][data_point_number])
                    ln7.set_data(xdata, ydata7)
                    ax.draw_artist(ln7)
                # 
                canvas.draw()
                plot_image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
                plot_image = plot_image.reshape(canvas.get_width_height()[::-1] + (4,))
            # 
            # 
            plot_image_resized = cv2.resize(plot_image, (target_height, target_height), interpolation=cv2.INTER_LINEAR)
            # plot_image_rgb = cv2.cvtColor(plot_image_resized, cv2.COLOR_RGBA2RGB)
            plot_image_bgr = cv2.cvtColor(plot_image_resized, cv2.COLOR_RGBA2BGR)
            # 
            if show_reasoning_traces:
                # white_box_width = frame_width // denom_  # Adjust width of the white box
                white_box_width = target_height  # Set white box width to match the height of the frame
                white_box = np.ones((frame_height, white_box_width, 3), dtype=np.uint8) * 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_color = (0, 0, 0)  
                thickness = text_thickness
                line_type = line_type
                text = final_reasoning_traces[frame_number]
                language_description_split = text.split(' ')
                language_description_wrapped = ''
                line_i_len = 0
                for i in range(len(language_description_split)):
                    if line_i_len < wrap_width:
                        language_description_wrapped += language_description_split[i] + ' '
                        line_i_len += len(language_description_split[i])
                    else:
                        language_description_wrapped += '\n' + language_description_split[i] + ' '
                        line_i_len = len(language_description_split[i])
                text = language_description_wrapped
                text = text.replace('</think><answer>', '</think>\n<answer>')
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (white_box_width - text_size[0]) // 2
                text_y = (frame_height + text_size[1]) // 2
                lines = text.split('\n')
                start_y = font_height 
                for i, line in enumerate(lines):
                    y = start_y + (i * font_height)  
                    res_ = cv2.putText(white_box, line, (10, y), font, font_scale, font_color, thickness, line_type)
                # 
                # combined_frame = np.hstack((frame, white_box, plot_image_rgb))
                combined_frame = np.hstack((frame, white_box, plot_image_bgr))
            else:
                # combined_frame = np.hstack((frame, plot_image_rgb))
                combined_frame = np.hstack((frame, plot_image_bgr))
            # 
            combined_frame = combined_frame.astype(np.uint8)
            # 
            if task_description is not None:
                # ---- create white top bar ----
                top_bar = np.ones((top_pad, combined_frame.shape[1], 3), dtype=np.uint8) * 255
                # ---- add text ----
                label = f'Task: "{task_description}"'
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                padding = 12
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = padding
                text_y = padding + text_h
                cv2.putText(
                    top_bar,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 0),  # black text
                    thickness,
                    cv2.LINE_AA,
                )
                # ---- stack on top of frame ----
                combined_frame = np.vstack((top_bar, combined_frame))
            # 
            assert combined_frame.shape[0] == output_height
            assert combined_frame.shape[1] == output_width
            assert combined_frame.dtype == np.uint8
            # out.write(combined_frame)
            out.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))
            frame_number += 1
            data_point_number += 1
        # 
        out.release()
    plt.close(fig)   
    del canvas 
    os.system(f"ffmpeg -y -i {plot_save_path} -c:v libx264 -pix_fmt yuv420p {plot_save_path.replace('.avi', '.mp4')}")
    os.remove(plot_save_path)



