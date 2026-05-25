#!/bin/bash
set -e
source /tmp/w8a8-venv/bin/activate
cd /nfs/FM/chenshuailin/projects/kernel_agents/slime
ORIG=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B
ROT=/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-rotated-mm-bf16

for variant in raw roundtrip_only fuse_bf16 fuse_fp32; do
  echo ============ variant=$variant ============
  python3 /tmp/rotation_probe.py --model-path $ORIG --variant $variant --output /tmp/probe_${variant}.pt
done

echo ============ variant=rotated_ckpt ============
python3 /tmp/rotation_probe.py --model-path $ROT --variant raw --output /tmp/probe_rotated.pt

echo ============ comparison ============
python3 /tmp/compare_rotation_probe.py \
  --baseline /tmp/probe_raw.pt \
  --variant /tmp/probe_roundtrip_only.pt \
  --variant /tmp/probe_fuse_bf16.pt \
  --variant /tmp/probe_fuse_fp32.pt \
  --variant /tmp/probe_rotated.pt
