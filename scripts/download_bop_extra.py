from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bop-benchmark/bop_extra",
    repo_type="dataset",  # Important: specify it's a dataset, not a model
    local_dir="/mnt/personal/jelint19/data/bop/",
)