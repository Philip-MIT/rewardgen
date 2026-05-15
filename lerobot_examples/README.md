
---
## Examples of annotating lerobot datasets 
```python
# pip install -U roboreason
import roboreason as rr

from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset_name = "jackvial/so101_pickplace_recap_merged_v2"
dataset = LeRobotDataset(dataset_name, video_backend="pyav")

task_description="Pick up the cube from the table and place it at the target location marked by the X."

# NOTE: SOLE-R1 is fastest, but all methods are slow with large datasets

# uses lerobot add_features function to add reward annotations to the dataset, stored in the '/with_reward' sub-directory
rr.annotate(model="robometer", lerobot_dataset=dataset, observation_name="observation.images.side", annotation_version="v1", task_description=task_description, num_reasoning_frames=10)
rr.annotate(model="sole-r1", lerobot_dataset=dataset, observation_name="observation.images.side", annotation_version="v1", task_description=task_description,  num_reasoning_frames=10)
rr.annotate(model="roboreward", lerobot_dataset=dataset, observation_name="observation.images.side", annotation_version="v1", task_description=task_description, num_reasoning_frames=10)
rr.annotate(model="topreward", lerobot_dataset=dataset, observation_name="observation.images.side", annotation_version="v1", task_description=task_description, num_reasoning_frames=10)
```


## Extracting the saved annotations and visualizing
```python
video_frames = rr.extract_frames(lerobot_dataset=dataset, observation_name="observation.images.side")
rewards_robometer = rr.extract_annotation(lerobot_dataset=dataset, model="robometer", annotation_version="v1")
rewards_sole = rr.extract_annotation(lerobot_dataset=dataset, model="sole-r1", annotation_version="v1")
rewards_roboreward = rr.extract_annotation(lerobot_dataset=dataset, model="roboreward", annotation_version="v1")
rewards_topreward = rr.extract_annotation(lerobot_dataset=dataset, model="topreward", annotation_version="v1")


plot_save_name_base = dataset_name.replace('/', '_')
plot_save_dir = f"../model_outputs/combined/lerobot/{plot_save_name_base}/"
for video_idx in range(len(video_frames)):
    output_combined = [
        {"model": "robometer", "rewards": rewards_robometer[video_idx]}, 
        {"model": "sole-r1", "rewards": rewards_sole[video_idx]}, 
        {"model": "roboreward", "rewards": rewards_roboreward[video_idx]}, 
        {"model": "topreward", "rewards": rewards_topreward[video_idx]}
    ]
    rr.video_plot(outputs=output_combined, plot_save_path=plot_save_dir + f"{plot_save_name_base}_episode_{video_idx}.mp4", video_frames=video_frames[video_idx], view_type='external', show_reasoning_traces=False, show_all_frames=False, fps_=10)

```


## Reading the saved rewards as pandas df
```python
import os
import pandas as pd
parquet_path = os.path.join(dataset.root / "with_reward" , "data/chunk-000/file-000.parquet")
df = pd.read_parquet(parquet_path)
df
#                                               action                                  observation.state  timestamp  frame_index  episode_index  index  task_index  rewards_robometer_v1  rewards_sole-r1_v1  rewards_roboreward_v1  rewards_topreward_v1  rewards_gemini-3-pro-preview_v1
# 0  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -97.12963, 97.00449, 86.924034, -2...   0.000000            0              0      0           0             14.105175            0.000000                    0.0              0.000000                         0.000000
# 1  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -97.12963, 97.00449, 86.924034, -2...   0.033333            1              0      1           0             14.292012            0.205128                    0.0              2.453601                         0.025641
# 2  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -96.94444, 97.20419, 86.176834, -2...   0.066667            2              0      2           0             14.478848            0.410256                    0.0              4.907203                         0.051282
# 3  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -96.01852, 97.60359, 83.93524, -24...   0.100000            3              0      3           0             14.665686            0.615385                    0.0              7.360804                         0.076923
# 4  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -95.833336, 97.60359, 80.69739, -2...   0.133333            4              0      4           0             14.852522            0.820513                    0.0              9.814405                         0.102564
# 5  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -96.57407, 97.60359, 79.20299, -24...   0.166667            5              0      5           0             15.039359            1.025641                    0.0             12.267995                         0.128205
# 6  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -96.296295, 97.60359, 78.45579, -2...   0.200000            6              0      6           0             15.226195            1.230769                    0.0             14.721596                         0.153846
# 7  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -95.74074, 97.60359, 77.95766, -24...   0.233333            7              0      7           0             15.413033            1.435897                    0.0             17.175198                         0.179487
# 8  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -95.64815, 97.60359, 77.58406, -24...   0.266667            8              0      8           0             15.599869            1.641026                    0.0             19.628799                         0.205128
# 9  [45.85219, -96.01361, 100.0, 76.41357, -24.210...  [45.420906, -95.92593, 97.60359, 77.33499, -24...   0.300000            9              0      9           0             15.786706            1.846154                    0.0             22.082401                         0.230769

```




