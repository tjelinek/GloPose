from pathlib import Path

from huggingface_hub import snapshot_download

dataset_name = "hot3d"
local_dir = Path("/mnt/personal/jelint19/data/bop")
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="bop-benchmark/datasets",
                  allow_patterns=f"{dataset_name}/*zip",
                  repo_type="dataset",
                  local_dir=local_dir)
