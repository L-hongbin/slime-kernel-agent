"""Export this workflow and a built tracer into an immutable diagnostic bundle."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / "tools/data/runtime_trace"
    target.mkdir(parents=True)
    for source in Path(__file__).parent.iterdir():
        if source.suffix in {".py", ".cu", ".cpp", ".h"} or source.name == "Makefile":
            shutil.copy2(source, target / source.name)
    shutil.copy2(args.native, args.output / "runtime_trace.so")
    manifest = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "bundle": str(args.output),
                "manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
