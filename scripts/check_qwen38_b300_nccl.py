"""Validate eight GPUs, TP4 training, TP2 rollout, and legacy TP2/CP2 communication groups."""

import datetime
import os

import torch
import torch.distributed as distributed


def check_group(ranks, group, rank, device):
    if rank not in ranks:
        return
    group_rank = ranks.index(rank)
    for dtype in (torch.float32, torch.bfloat16):
        for count in (1, 1024 * 1024):
            values = torch.full((count,), rank + 1, dtype=dtype, device=device)
            distributed.all_reduce(values, group=group)
            torch.testing.assert_close(values, torch.full_like(values, sum(member + 1 for member in ranks)))
        source = torch.full((1024,), rank + 1, dtype=dtype, device=device)
        gathered = torch.empty(len(ranks) * source.numel(), dtype=dtype, device=device)
        distributed.all_gather_into_tensor(gathered, source, group=group)
        for index, member in enumerate(ranks):
            torch.testing.assert_close(gathered.chunk(len(ranks))[index], torch.full_like(source, member + 1))
        reduced = torch.empty_like(source)
        distributed.reduce_scatter_tensor(reduced, gathered, group=group)
        torch.testing.assert_close(reduced, torch.full_like(reduced, (rank + 1) * len(ranks)))
        exchanged = torch.empty_like(gathered)
        distributed.all_to_all_single(exchanged, gathered, group=group)
        torch.testing.assert_close(exchanged, torch.full_like(exchanged, rank + 1))
    if len(ranks) == 2:
        peer = ranks[1 - group_rank]
        source = torch.full((4096,), rank + 1, device=device)
        destination = torch.empty_like(source)
        requests = distributed.batch_isend_irecv(
            [
                distributed.P2POp(distributed.isend, source, peer, group),
                distributed.P2POp(distributed.irecv, destination, peer, group),
            ]
        )
        for request in requests:
            request.wait()
        torch.testing.assert_close(destination, torch.full_like(destination, peer + 1))
    torch.cuda.synchronize(device)


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    distributed.init_process_group("nccl", timeout=datetime.timedelta(seconds=120), device_id=device)
    if distributed.get_world_size() != 8:
        raise ValueError("Run with torchrun --standalone --nproc-per-node=8")
    layouts = [list(range(8)), [0, 1, 2, 3], [4, 5, 6, 7], [0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3]]
    groups = []
    for ranks in layouts:
        group = distributed.new_group(ranks, timeout=datetime.timedelta(seconds=120))
        groups.append((ranks, group))
        check_group(ranks, group, rank, device)
        distributed.barrier()
        if rank == 0:
            print(f"PASS: ranks={ranks}, BF16/FP32 collectives and paired P2P", flush=True)
    for ranks, group in reversed(groups):
        if rank in ranks:
            distributed.destroy_process_group(group)
    distributed.destroy_process_group()
    if rank == 0:
        print(f"PASS: NCCL {torch.cuda.nccl.version()}, PyTorch {torch.__version__}", flush=True)


if __name__ == "__main__":
    main()
