#!/usr/bin/env python3
"""Validate inputs and outputs for the authorized final GEPA-V2 evaluation."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / 'experiments/qwen38_b300_baseline_t1_nocov_noprs'
SUITE = EXP / 'eval_kernelbench_gepav2'
ORIGIN = Path('/nfs/hw-data/ms/FM/checkpoints/Qwen-Zoo/Qwen3.8-27B')
HF = EXP / 'hf/iter_0000079'
TRAIN_JOB = 'qwen38_b300_baseline_t1_nocov_noprs'
EXPECTED = {1: 100, 2: 100, 3: 50}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def audit_data():
    import pyarrow.parquet as pq
    manifest = json.loads((SUITE / 'provenance/datasets.json').read_text())
    reports, contents = [], []
    for level, count in EXPECTED.items():
        item = next(x for x in manifest if x['level'] == level)
        path = SUITE / 'data' / f'kernelbench_level{level}_val.parquet'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        rows = pq.read_table(path).to_pylist()
        assert len(rows) == count, (level, len(rows))
        roles = set()
        for row in rows:
            prompt = row['prompt']
            assert isinstance(prompt, list) and prompt
            assert all(isinstance(m['content'], str) and m['content'].strip() for m in prompt)
            roles.add(tuple(m['role'] for m in prompt))
            text = '\n'.join(m['content'] for m in prompt)
            assert text.startswith('Optimize the provided PyTorch `Model`')
            assert 'TVM-FFI' in text
            contents.append(text)
        reports.append({**item, 'rows': count, 'roles': sorted(roles),
                        'candidates_per_prompt': 8, 'expected_trajectories': count * 8})
    prefix = os.path.commonprefix(contents)
    assert len(prefix) > 100, len(prefix)
    (SUITE / 'provenance/gepav2_shared_instruction.txt').write_text(prefix)
    report = {'datasets': reports, 'shared_initial_instruction_chars': len(prefix),
              'shared_initial_instruction_sha256': hashlib.sha256(prefix.encode()).hexdigest(),
              'note': 'Use complete source prompt messages unchanged; do not substitute only a response template.'}
    write_json(SUITE / 'provenance/prompt_audit.json', report)
    print(json.dumps({'rows': EXPECTED, 'shared_instruction_chars': len(prefix), 'total_trajectories': 2000}))


def checkpoint_info(path):
    from torch.distributed.checkpoint import FileSystemReader
    assert (path / '.metadata').is_file(), path
    metadata = FileSystemReader(str(path)).read_metadata()
    shards = set()
    for info in metadata.storage_data.values():
        shard = path / info.relative_path
        assert shard.is_file() and shard.stat().st_size >= info.offset + info.length, shard
        shards.add(shard)
    assert shards and metadata.state_dict_metadata
    return {'path': str(path), 'shards': len(shards), 'bytes': sum(p.stat().st_size for p in shards),
            'state_entries': len(metadata.state_dict_metadata)}


def check_training():
    from ray.job_submission import JobSubmissionClient
    job = JobSubmissionClient('http://127.0.0.2:8270').get_job_info(TRAIN_JOB)
    assert str(job.status) == 'SUCCEEDED', f'Training not complete: {job.status}'
    latest = int((EXP / 'checkpoints/latest_checkpointed_iteration.txt').read_text().strip())
    assert latest == 79, latest
    result = {'job_id': TRAIN_JOB, 'status': str(job.status), 'end_time': job.end_time,
              'iteration': latest, 'checkpoint': checkpoint_info(EXP / f'checkpoints/iter_{latest:07d}')}
    write_json(SUITE / 'provenance/training_complete.json', result)
    print(json.dumps(result))


def tensor_index(root):
    return json.loads((root / 'model.safetensors.index.json').read_text())


def inspect_hf(root):
    from safetensors import safe_open
    index, reference = tensor_index(root), tensor_index(ORIGIN)
    assert set(index['weight_map']) == set(reference['weight_map']), 'HF tensor key mismatch'
    def inventory(folder, mapping):
        grouped = defaultdict(list)
        for key, shard in mapping.items():
            grouped[shard].append(key)
        result = {}
        for shard, keys in grouped.items():
            path = folder / shard
            assert path.is_file() and path.stat().st_size > 0, path
            with safe_open(str(path), framework='pt', device='cpu') as reader:
                for key in keys:
                    shape = reader.get_slice(key)
                    result[key] = (shape.get_shape(), shape.get_dtype())
        return result
    actual = inventory(root, index['weight_map'])
    original = inventory(ORIGIN, reference['weight_map'])
    assert actual == original, 'HF shape or dtype mismatch'
    for asset in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        assert (root / asset).is_file(), asset
    return {'path': str(root), 'tensors': len(actual), 'shards': len(set(index['weight_map'].values())),
            'total_tensor_bytes': index['metadata']['total_size']}


def complete_hf():
    from safetensors import safe_open
    from safetensors.torch import save_file
    assert (SUITE / 'provenance/training_complete.json').is_file(), 'Verify training success first'
    index, reference = tensor_index(HF), tensor_index(ORIGIN)
    missing = set(reference['weight_map']) - set(index['weight_map'])
    assert not (set(index['weight_map']) - set(reference['weight_map']))
    assert all(key.startswith('model.visual.') for key in missing), sorted(missing)[:10]
    if missing:
        tensors = {}
        by_shard = defaultdict(list)
        for key in missing:
            by_shard[reference['weight_map'][key]].append(key)
        for shard, keys in by_shard.items():
            with safe_open(str(ORIGIN / shard), framework='pt', device='cpu') as reader:
                for key in keys:
                    tensors[key] = reader.get_tensor(key)
        name = 'model-origin-visual.safetensors'
        temp = HF / (name + '.tmp')
        save_file(tensors, str(temp))
        temp.replace(HF / name)
        index['weight_map'].update({key: name for key in tensors})
        index['metadata']['total_size'] += sum(t.numel() * t.element_size() for t in tensors.values())
        write_json(HF / 'model.safetensors.index.json', index)
    result = inspect_hf(HF)
    result['restored_frozen_visual_tensors'] = len(missing)
    write_json(SUITE / 'provenance/hf_complete.json', result)
    print(json.dumps(result))


def check_results(level):
    import torch
    from examples.kernel_agent.eval.summarize_eval import summarize
    root = SUITE / f'level{level}'
    dumps = list((root / 'dumps').rglob('eval_*.pt'))
    assert len(dumps) == 1, f'Expected one final eval dump, found {dumps}'
    obj = torch.load(dumps[0], map_location='cpu', weights_only=False)
    samples = obj.get('samples', []) if isinstance(obj, dict) else obj
    assert len(samples) == EXPECTED[level] * 8, len(samples)
    result = summarize(samples, (1.0, 1.2))
    assert result['missing_env_result'] == 0, 'Missing env_result: inspect infrastructure before accepting scores'
    result.update(level=level, dump=str(dumps[0]), expected_trajectories=EXPECTED[level] * 8)
    write_json(root / 'validated_summary.json', result)
    print(json.dumps(result))
    paths = [SUITE / f'level{i}/validated_summary.json' for i in EXPECTED]
    if all(p.is_file() for p in paths):
        write_json(SUITE / 'validated_summary.json', [json.loads(p.read_text()) for p in paths])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('audit-data', 'check-training', 'check-checkpoint', 'complete-hf', 'validate-hf', 'check-results'))
    parser.add_argument('--path', type=Path)
    parser.add_argument('--level', type=int, choices=tuple(EXPECTED))
    args = parser.parse_args()
    if args.command == 'audit-data': audit_data()
    elif args.command == 'check-training': check_training()
    elif args.command == 'check-checkpoint': print(json.dumps(checkpoint_info(args.path)))
    elif args.command == 'complete-hf': complete_hf()
    elif args.command == 'validate-hf': print(json.dumps(inspect_hf(args.path or HF)))
    elif args.command == 'check-results':
        assert args.level is not None
        check_results(args.level)
