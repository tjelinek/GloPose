from pathlib import Path
from huggingface_hub import snapshot_download

dataset_name = "hot3d"
local_dir = Path(f"/mnt/personal/jelint19/data/bop/{dataset_name}")
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=f"bop-benchmark/{dataset_name}",
    allow_patterns=["test_aria/*.tar"],
    repo_type="dataset",
    local_dir=local_dir
)
