# SPEC

## Training/Rollout Node

- `ssh -p 23452 root@192.168.16.67` (already a started container)
- GPUs: 8 x NVIDIA A800-SXM4-80GB

## DrKernel Data

- Slime-format train parquet: `data/drkernel-rl-data-0513/train.parquet`
- Slime-format KernelBench L1 validation parquet: `data/kernelbench-level1-validation/train.parquet`
- Review prompt samples are written to `checkpoints/drkernel_prompt_debug/formatted_prompt_examples.txt`.

## Current Qwen3.5-9B Debug Run

- Script: `scripts/debug.sh`
- Latest successful log: `checkpoints/Qwen3.5-9B/run_20260515_122920.log`
- Latest Ray job: `raysubmit_v7whRX3AfMxtL3YY` succeeded on 2026-05-15 13:13.
- SGLang context length: `32768`
- SGLang max running requests: `64`
- SGLang static memory fraction: `0.7`
- Eval result: `eval/kernelbench_level1 = 0.035` on 800 samples; response length mean `9750.64`, max `31814`, truncated ratio `0.03125`.
- Observed SGLang behavior: `--sglang-context-length 32768` and `--sglang-max-running-requests 64` are received by SGLang, but Qwen3.5-9B still preallocates large KV/Mamba cache.

## Docker Environment

- Active container was started with: `docker run -itd --gpus all -p 23452:22 --device /dev/infiniband -v /nfs:/nfs --ipc host --ulimit memlock=-1 --ulimit stack=67108864 --privileged --name csl_slime 192.168.14.129:80/library/slime:nightly-dev-20260430b bash`
- Image: `192.168.14.129:80/library/slime:nightly-dev-20260430b`
- Image digest observed on pull: `sha256:88ceed1788eb19272418d493ed4f23d7a9dd06d139c4d9ee5fe6ffb8bc938cb1`
- `bash set_env.sh` has been run inside `csl_slime` from the slime repo root.
