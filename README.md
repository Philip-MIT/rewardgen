<div align="center">


# RewardGen 

**RewardGen** is a python package that makes it easy to apply any ***reward model*** to your robot videos and plot the rewards as shown below.
(All example videos at: https://philip-mit.github.io/rewardgen_view/)

</div>

## Examples

<!-- https://github.com/user-attachments/assets/cd481f28-0cb3-4874-bd50-1ec3ad8326ec -->
https://github.com/user-attachments/assets/3c444096-d3dd-47c7-b09d-90b0756d0f72



## Supported Models
- Robometer (https://robometer.github.io)
- SOLE-R1 (https://philip-mit.github.io/sole-r1/)
- TOPReward (https://topreward.github.io/webpage/)
- RoboReward (https://arxiv.org/abs/2601.00675)
- OpenAI models (e.g., `"GPT-5"`)
- Google models (e.g., `"Gemini-3.1-Pro-Preview"`)

## ToDos
- [ ] Enable fine-tuning of reward models on new datasets/demonstrations

## File Structure

```
rewardgen/
├── rewardgen/         # Main package
│   ├── robometer/         # Robometer code
│   ├── sole.py            # SOLE-R1 code
│   ├── roboreward.py      # RoboReward code
│   ├── topreward.py       # TOPReward code
│   └── api_models.py      # OpenAI and Gemini APIs
├── test_videos/        # Example videos to test
├── model_outputs/      # Example videos showing model outputs
├── docs/   
│   ├── lerobot_dataset_reward_annotation.mdx  # Examples showing integration with lerobot datasets
└── pyproject.toml      # Dependencies (uv)
```


## Install
### Option 1: quick pip install
```bash
pip install -U rewardgen
```

### Option 2: use [uv](https://github.com/astral-sh/uv) for dependency management
```bash
# 1) Clone the repository
git clone https://github.com/Philip-MIT/rewardgen

# 2) Install `uv`
pip install uv

# 3) Sync environment
cd rewardgen
uv sync

# 4) Activate environment
source .venv/bin/activate
```


---

## Optional: Pre-download model checkpoints
```bash

# SOLE-R1 (8B) 
python -c "from rewardgen.utils.model_utils import get_model_dir; get_model_dir('sole-r1')"

# Robometer (4B)
python -c "from rewardgen.utils.model_utils import get_model_dir; get_model_dir('robometer')"

# TOPReward (based on Qwen3-VL-8B)
python -c "from rewardgen.utils.model_utils import get_model_dir; get_model_dir('topreward')"

# RoboReward (8B)
python -c "from rewardgen.utils.model_utils import get_model_dir; get_model_dir('roboreward')"

> **Note:** Robometer is ~8GB. SOLE-R1, RoboReward, and TOPReward are ~17GB each.

```
## Optional: Download all test videos and example model outputs
```bash
# 1) Install gcloud: https://cloud.google.com/sdk/docs/install

# 2) Go to target directory
# cd /path/to/rewardgen

# Optional: disable credentials so you don't have to authenticate
gcloud config set auth/disable_credentials True

# Download test videos
gcloud storage cp --recursive gs://roboreason-view-videos-philip/test_videos ./

# Download model outputs for all test videos
gcloud storage cp --recursive gs://roboreason-view-videos-philip/model_outputs ./

# Optional: re-enable credentials afterward if you disabled them above.
gcloud config set auth/disable_credentials False

```

---
## Quick start: Example reward generation and plotting
```python
# pip install -U rewardgen
from rewardgen import generate, video_plot

video_view_external_paths = ['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_external.mp4']
video_view_wrist_paths = ['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_wrist.mp4']
task_description="Pick up the cube from the table."

# Robometer
response = generate(model="Robometer",  task_description=task_description, video_view_external_paths=video_view_external_paths,  verbose=False)
print(response.rewards)
print(response.success_probs)
output_robometer = {"model": "Robometer", "rewards": response.rewards[0]}

# SOLE-R1
response = generate(model="SOLE-R1",  task_description=task_description, video_view_external_paths=video_view_external_paths, video_view_wrist_paths=video_view_wrist_paths, verbose=False)
print(response.rewards)
print(response.output_text)
output_sole = {"model": "SOLE-R1", "rewards": response.rewards[0], "output_text": response.output_text[0]}

# Optional: Ground-truth rewards (available for test videos from sim environments)
import json
with open('test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/data.json', 'r') as f:
    data = json.load(f)

output_groundtruth = {"model": "Ground truth", "rewards": data['ground-truth rewards']}

# Plot
video_plot(outputs=[output_groundtruth, output_sole, output_robometer], plot_save_path='model_outputs/combined/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4', video_view_external_path=video_view_external_paths[0], video_view_wrist_path=video_view_wrist_paths[0], task_description=task_description)

```

---
## Examples for generating across all models

### Robometer
```python

from rewardgen import generate

video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_external.mp4']
task_description="Pick up the cube from the table."

response = generate(
    model="Robometer",  
    task_description=task_description, 
    video_paths=video_paths, 
    view_type="external",
    verbose=False
)
print(response.rewards)
print(response.success_probs)

```

### SOLE-R1
```python

from rewardgen import generate

video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4']
task_description="Pick up the cube from the table."

response = generate(
    model="SOLE-R1",  
    task_description=task_description, 
    video_paths=video_paths, 
    view_type='external and wrist',
    verbose=False
)
print(response.rewards)
print(response.output_text)

output_sole = {"model": "SOLE-R1", "rewards": response.rewards[0], "output_text": response.output_text[0]}

# Plotting with show_output_text=True
video_plot(
    outputs=[output_sole], 
    plot_save_path='model_outputs/combined/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4', 
    video_path=video_paths[0],
    show_output_text=True,
    task_description=task_description,
    verbose=False
)
```


### TOPReward
```python

from rewardgen import generate

video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_external.mp4']
task_description="Pick up the cube from the table."

response = generate(
    model="TOPReward",  
    task_description=task_description, 
    video_paths=video_paths, 
    view_type='external',
    verbose=False
)
print(response.rewards)

```

### RoboReward
```python

from rewardgen import generate

video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_external.mp4']
task_description="Pick up the cube from the table."

response = generate(
    model="RoboReward",  
    task_description=task_description, 
    video_paths=video_paths, 
    view_type='external',
    verbose=False
)
print(response.rewards)

```

### GPT-5 (and other OpenAI models)
```python

from rewardgen import generate

video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_external.mp4']
task_description="Pick up the cube from the table."

# requires OpenAI API key: https://developers.openai.com/api/docs/quickstart
API_KEY = "..."

response = generate(
    model="GPT-5",  
    task_description=task_description, 
    video_paths=video_paths, 
    view_type='external', 
    key=API_KEY, 
    verbose=False
)
print(response.rewards)
print(response.output_text)
```

### Gemini-3-Pro (and other Google models)
```python

from rewardgen import generate

video_paths=['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_external.mp4']
task_description="Pick up the cube from the table."

# requires Gemini API key: https://ai.google.dev/gemini-api/docs/api-key
API_KEY = "..."

response = generate(
    model="Gemini-3.1-Pro-Preview",  
    task_description=task_description, 
    video_paths=video_paths, 
    view_type='external', 
    key=API_KEY,
    verbose=False
)
print(response.rewards)
print(response.output_text)
```

## Video plotting
```python

from rewardgen import generate, video_plot

video_view_external_paths = ['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_external.mp4']
video_view_wrist_paths = ['test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/view_wrist.mp4']
task_description="Pick up the cube from the table."

# Robometer
response = generate(model="Robometer",  task_description=task_description, video_view_external_paths=video_view_external_paths,  verbose=False)
output_robometer = {"model": "Robometer", "rewards": response.rewards[0]}

# SOLE-R1
response = generate(model="SOLE-R1",  task_description=task_description, video_view_external_paths=video_view_external_paths, video_view_wrist_paths=video_view_wrist_paths, verbose=False)
output_sole = {"model": "SOLE-R1", "rewards": response.rewards[0], "output_text": response.output_text[0]}

# Optional: Ground-truth rewards (available for test videos from sim environments)
import json
with open('test_videos/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37/data.json', 'r') as f:
    data = json.load(f)

output_groundtruth = {"model": "Ground truth", "rewards": data['ground-truth rewards']}

video_plot(
    outputs=[output_groundtruth, output_sole, output_robometer], 
    plot_save_path='model_outputs/combined/robosuite/lift/unsuccessful/robosuite_lift_episode_11_unsuccessful_max_reward_37.mp4', 
    video_view_external_path=video_view_external_paths[0], 
    video_view_wrist_path=video_view_wrist_paths[0],
    task_description=task_description,
    verbose=False
)
```

## Reward generation and plotting across many videos
```python

from rewardgen import generate
import glob
import json

video_paths = glob.glob('test_videos/robosuite/lift/unsuccessful/*.mp4')
task_description="Pick up the cube from the table."

## REWARD GENERATION
# SOLE-R1 for all videos
response = generate(model="SOLE-R1",  task_description=task_description, video_paths=video_paths, view_type='external and wrist')

## PLOTTING
plot_save_dir = 'model_outputs/sole-r1/'
for video_idx in range(len(video_paths)):
    output_sole = {"model": "SOLE-R1", "rewards": response.rewards[video_idx]}
    # Optional: Ground-truth rewards (available for test videos from sim environments)
    with open(video_paths[video_idx].replace(".mp4", "/data.json"), 'r') as f:
        data = json.load(f)
    
    output_groundtruth = {"model": "Ground truth", "rewards": data['ground-truth rewards']}
    video_plot(
        outputs = [output_groundtruth, output_sole], 
        plot_save_path = plot_save_dir + video_paths[video_idx].split('test_videos/')[-1] , 
        video_path = video_paths[video_idx],
        task_description=task_description,
        verbose = False
    )
```



---



---
## Acknowledgements
RewardGen builds upon the following repos: 
- TOPReward (https://github.com/TOPReward/TOPReward)
- Robometer (https://github.com/robometer/robometer)
- RewardScope (https://github.com/philfung/reward-scope)

Also thank you to [Jack Vial](https://github.com/jackvial) for the SO-101 videos.


