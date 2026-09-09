"""Tensor storage leases; argument presence records identity, never read/write."""

import ctypes
import json
import weakref
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode


class StorageObserver(TorchDispatchMode):
    def __init__(self, traced=False, checkpoint=None, max_leases=None):
        super().__init__()
        self.leases = []
        self.live = {}
        self.native = ctypes.CDLL(None) if traced else None
        if self.native:
            self.native.coarse_position.restype = ctypes.c_uint64
        self.host_position = 0
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.max_leases = max_leases
        self.unknowns = set()

    def position(self):
        return self.native.coarse_position() if self.native else self.host_position

    def observe(self, value, role=None, start=None):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            storage = value.untyped_storage()
            key = storage._cdata
            entry = self.live.get(key)
            if entry is None:
                if self.max_leases is not None and len(self.leases) >= self.max_leases:
                    self.unknowns.add("storage_registry_truncated")
                    return
                record = {
                    "id": f"storage:{len(self.leases)}",
                    "base": storage.data_ptr(),
                    "bytes": storage.nbytes(),
                    "first_seq": self.position() if start is None else start,
                    "last_seq": None,
                    "roles": [],
                    "views": [],
                    "role_views": {},
                }
                self.leases.append(record)

                def released(_ref, key=key, record=record):
                    record["last_seq"] = self.position()
                    self.live.pop(key, None)

                self.live[key] = (weakref.ref(storage, released), record)
            else:
                record = entry[1]
            if role and role not in record["roles"]:
                record["roles"].append(role)
            view = {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "offset_bytes": value.storage_offset() * value.element_size(),
                "dtype": str(value.dtype),
                "element_size": value.element_size(),
            }
            if role:
                record["role_views"][role] = view
            if view not in record["views"]:
                record["views"].append(view)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                self.observe(item, f"{role}:{i}" if role else None, start)
        elif isinstance(value, dict):
            for key, item in value.items():
                self.observe(item, f"{role}:{key}" if role else None, start)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        start = self.position()
        self.observe(args, start=start)
        self.observe(kwargs or {}, start=start)
        self.save_checkpoint()
        result = func(*args, **(kwargs or {}))
        self.observe(result, start=start)
        self.host_position += 1
        self.save_checkpoint()
        return result

    def save_checkpoint(self):
        if self.checkpoint is not None:
            temporary = self.checkpoint.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.as_dict()))
            temporary.replace(self.checkpoint)

    def as_dict(self):
        return {
            "schema": "storage-leases/v1",
            "allocations": self.leases,
            "identity_source": "weak storage lifetime plus CUDA storage address; views share identity",
            "argument_presence_is_read_write": False,
            "holds_tensor_values_alive": False,
            "unknowns": sorted(self.unknowns),
        }
