
<div align="center">


# RoboReason 

**RoboReason** is a python package that makes it easy to apply any ***reward model*** or ***video-language reasoning model*** to your robot videos.

</div>

## Supported Models
- Robometer (https://robometer.github.io)
- TOPReward (https://topreward.github.io/webpage/)
- RoboReward (https://arxiv.org/abs/2601.00675)
- SOLE-R1 (https://philipmit.github.io/sole-r1/)
- OpenAI models (e.g., `"gpt-5"`)
- Google models (e.g., `"gemini-3-pro-preview"`)

## ToDos
- [ ] Enable fine-tuning of reward models on custom datasets

## 📦 File Structure

```
roboreason/
├── roboreason/         # Main package
│   ├── robometer/         # Robometer code
│   ├── sole.py            # SOLE-R1 code
│   ├── roboreward.py      # RoboReward code
│   ├── topreward.py       # TOPReward code
│   └── api_models.py      # OpenAI and Gemini APIs
├── test_videos/        # Example videos to test
├── model_outputs/      # Videos showing model outputs
├── lerobot_examples/   # Examples showing integration with lerobot datasets
└── pyproject.toml      # Dependencies (uv)
```


## Install
### Option 1: quick pip install
```bash
pip install -U roboreason
```

### Option 2: use [uv](https://github.com/astral-sh/uv) for dependency management

#### 1. Clone the repository:
```bash
git clone https://github.com/philipmit/roboreason
```

#### 2. Install `uv`
```bash
pip install uv
```

#### 3. Sync environment
```bash
uv sync
```

#### 4. Activate environment
```bash
source .venv/bin/activate
```

---

## Pre-download model checkpoints (optional)
```bash

# SOLE-R1 (8B)
python -c "from roboreason.utils.model_utils import get_model_dir; get_model_dir('sole')"

# Robometer (4B)
python -c "from roboreason.utils.model_utils import get_model_dir; get_model_dir('robometer')"

# TOPReward (based on Qwen3-VL-8B)
python -c "from roboreason.utils.model_utils import get_model_dir; get_model_dir('topreward')"

# RoboReward (8B)
python -c "from roboreason.utils.model_utils import get_model_dir; get_model_dir('roboreward')"

```
## Download all test videos from google drive [link here](https://drive.google.com/drive/folders/1tZndEwzfdyUY6Rh35oV1rAhpWqVzA2fQ?usp=sharing) (optional)
```bash

# pip install gdown
# cd /path/to/roboreason/test_videos/
gdown https://drive.google.com/drive/folders/1pXmiN-l8-khC4WABoMAn6saDvHjHGub0?usp=sharing

```
## Download all videos showing example model outputs from google drive [link here](https://drive.google.com/drive/folders/1tZndEwzfdyUY6Rh35oV1rAhpWqVzA2fQ?usp=sharing) (optional)
```bash

# pip install gdown
# cd /path/to/roboreason/model_outputs/
gdown https://drive.google.com/drive/folders/1gi-sTk8JssO9_UO6dHTqnkeAMjyW0YZb?usp=sharing

```
---
## Quick start: Example reward generation and plotting
```python
# pip install -U roboreason
import roboreason as rr

video_paths = ['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4']
task_description="Pick up the cube from the table."

# Robometer
rewards, success_probs = rr.generate(model="robometer",  task_description=task_description, video_paths=video_paths, view_type_per_video=['external'])
output_robometer = {"model": "robometer", "rewards": rewards[0]}

# SOLE-R1
rewards, reasoning_traces = rr.generate(model="sole-r1",  task_description=task_description, video_paths=video_paths, view_type_per_video=['external and wrist'])
output_sole = {"model": "sole-r1", "rewards": rewards[0], "reasoning_traces": reasoning_traces[0]}

rr.video_plot(outputs=[output_sole, output_robometer], plot_save_path='model_outputs/combined/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4', video_path = video_paths[0])

```

---
## Examples for generating across all models

### Robometer
```python

import roboreason as rr

rewards, success_probs = rr.generate(
    model="robometer",  
    task_description="Pick up the cube from the table.", 
    video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4'], 
    view_type_per_video=['external']
)

```

### SOLE-R1
```python

import roboreason as rr

rewards, reasoning_traces = rr.generate(
    model="sole-r1",  
    task_description="Pick up the cube from the table.", 
    video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4'], 
    view_type_per_video=['external and wrist']
)
```


### TOPReward
```python

import roboreason as rr

rewards = rr.generate(
    model="topreward",  
    task_description="Pick up the cube from the table.", 
    video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4'], 
    view_type_per_video=['external']
)

```

### RoboReward
```python

import roboreason as rr

rewards = rr.generate(
    model="roboreward",  
    task_description="Pick up the cube from the table.", 
    video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4'], 
    view_type_per_video=['external']
)

```

### GPT-5 (and other OpenAI models)
```python

import roboreason as rr

# requires OpenAI API key: https://developers.openai.com/api/docs/quickstart
API_KEY = "..."

rewards, reasoning_traces = rr.generate(
    model="gpt-5",  
    task_description="Pick up the cube from the table.", 
    video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4'], 
    view_type_per_video=['external'], 
    key=API_KEY
)
```

### Gemini-3-Pro (and other Google models)
```python

import roboreason as rr

# requires Gemini API key: https://ai.google.dev/gemini-api/docs/api-key
API_KEY = "..."

rewards, reasoning_traces = rr.generate(
    model="gemini-3-pro-preview",  
    task_description="Pick up the cube from the table.", 
    video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4'], 
    view_type_per_video=['external'], 
    key=API_KEY
)
```

## Video plotting
```python

import roboreason as rr

# Robometer
rewards, success_probs = rr.generate(model="robometer",  task_description=task_description, video_paths=video_paths, view_type_per_video=['external'])
output_robometer = {"model": "robometer", "rewards": rewards[0]}

# SOLE-R1
rewards, reasoning_traces = rr.generate(model="sole-r1",  task_description=task_description, video_paths=video_paths, view_type_per_video=['external and wrist'])
output_sole = {"model": "sole-r1", "rewards": rewards[0], "reasoning_traces": reasoning_traces[0]}

rr.video_plot(
    outputs=[output_sole, output_robometer], 
    plot_save_path='model_outputs/combined/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4', 
    video_path = 'test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4'
)
```


---


## rr.generate

| Argument              | Type        | Required | Description                                                                                                                                    |
| --------------------- | ----------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`               | `str`       | ✅        | Name of the model to use. Options include: `"robometer"`, `"sole-r1"`, `"topreward"`, `"roboreward"`, OpenAI models (e.g.`"gpt-5"`), Google models (e.g., `"gemini-3-pro-preview"`) |
| `task_description`    | `str`       | ✅        | Natural language description of the task the robot is performing.                                                                              |
| `video_paths`         | `List[str]` | ✅        | List of paths to input video files.                                                                                                            |
| `view_type_per_video` | `List[str]` | ✅        | List specifying the camera view(s) used for reward reasoning for each video (e.g., `"external"`, `"wrist"`, or `"external and wrist"`).                                  |
| `key`                 | `str`       | ❌        | API key required for external models (e.g., OpenAI or Gemini). Not needed for local models.                                                    |


| Model Type             | Return Values               |
| ---------------------- | --------------------------- |
| SOLE-R1 / GPT / Gemini | `rewards, reasoning_traces` |
| Robometer              | `rewards, success_probs`    |
| TOPReward / RoboReward | `rewards`                   |


## rr.video_plot

| Argument                | Type         | Required | Description                                                                               |
| ----------------------- | ------------ | -------- | ----------------------------------------------------------------------------------------- |
| `outputs`               | `List[dict]` | ❌*       | List of model outputs (e.g., from `rr.generate`) to visualize together.                   |
| `plot_save_path`        | `str`        | ❌        | Path where the output video with overlays will be saved.                                  |
| `video_path`            | `str`        | ❌        | Path to the original video file being visualized.                                         |
| `view_type`             | `str`        | ❌        | View type used for visualization (e.g., `"external"`, `"wrist"`, `"external and wrist"`). |
| `show_reasoning_traces` | `bool`       | ❌        | Whether to overlay reasoning traces on the video. Default: `False`.                       |
| `show_all_frames`       | `bool`       | ❌        | Whether to render all frames instead of sampled frames. Default: `False`.                 |
| `model`                 | `str`        | ❌**      | Model name (used when calling `video_plot` directly instead of passing `outputs`).        |
| `task_description`      | `str`        | ❌**      | Task description (used in direct-call mode).                                              |
| `video_paths`           | `List[str]`  | ❌**      | Input videos (used in direct-call mode).                                                  |
| `view_type_per_video`   | `List[str]`  | ❌**      | View types per video (used in direct-call mode).                                          |
| `key`                   | `str`        | ❌**      | API key (if required for model).                                                          |








