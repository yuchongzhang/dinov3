from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='ILSVRC/imagenet-1k',
    repo_type='dataset',
    local_dir='/home/yucz/projects/aip-papyan/shared/imagenet-1k',
)