

import os
from pathlib import Path
from huggingface_hub import snapshot_download

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "roboreason"

MODEL_REGISTRY = {
    "roboreward": "teetone/RoboReward-8B",
    "robometer": "robometer/Robometer-4B",
    "topreward": "Qwen/Qwen3-VL-8B-Instruct",
    "sole": "pschro/SOLE-R1-8B",
}


def get_model_dir(model_key: str, user_path: str | None = None) -> str:
    """
    Resolve local path to model. Download if needed.
    Priority:
    1. user_path argument
    2. environment variable
    3. default cache dir (~/.cache/roboreason)
    """
    # 
    # 1. explicit user override
    if user_path is not None:
        return user_path
    # 
    # 2. env var override
    env_var = f"ROBOREASON_{model_key.upper()}_PATH"
    if env_var in os.environ:
        return os.environ[env_var]
    # 
    # 3. default cache location
    model_id = MODEL_REGISTRY[model_key]
    local_dir = DEFAULT_CACHE_DIR / model_id.replace("/", "_")
    # 
    if not local_dir.exists():
        print(f"[RoboReason] Downloading {model_id} to {local_dir} ...")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
    # 
    return str(local_dir)

