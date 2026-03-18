import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="hot3d")
parser.add_argument("--include", nargs="*", default=None,
                    help="Glob patterns to include, e.g. 'train_aria/*.tar' 'models/**'")
args = parser.parse_args()

local_dir = Path(f"/mnt/data/vrg/public_datasets/bop/{args.dataset}")
local_dir.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=f"bop-benchmark/{args.dataset}",
    allow_patterns=args.include,
    repo_type="dataset",
    local_dir=local_dir,
)
