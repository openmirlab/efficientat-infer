"""CLI for one EfficientAT AudioSet inference call.

Outputs complete native scores and metadata as JSON. Model acquisition and
audio handling stay in the package session, not duplicated here.
Reads: CLI arguments and the public EfficientATSession facade.
"""

import argparse
import json
import sys
from pathlib import Path

from .session import EfficientATSession


def main(argv=None):
    parser = argparse.ArgumentParser(prog="efficientat-infer")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", choices=("mn04", "mn10"), default="mn10")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pad-to-seconds", type=float)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    with EfficientATSession(
        args.model, device=args.device, cache_dir=args.cache_dir,
        checkpoint_dir=args.checkpoint_dir, checkpoint_config=args.checkpoint_config,
        offline=args.offline,
    ) as session:
        session.load()
        result = session.infer(args.audio, pad_to_seconds=args.pad_to_seconds)
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
