"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import evaluate_protocol
from .profiles import PROFILES
from .protocol import ProtocolError, load_protocol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="complexmd-eval")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate a protocol and its input paths")
    validate.add_argument("protocol", type=Path)
    validate.add_argument("--profile", choices=sorted(PROFILES))
    validate.add_argument("--no-check-paths", action="store_true")
    run = sub.add_parser("run", help="run an evaluation")
    run.add_argument("protocol", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--profile", choices=sorted(PROFILES))
    profiles = sub.add_parser("profiles", help="show built-in profiles")
    profiles.set_defaults(show_profiles=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "profiles":
            print(json.dumps(PROFILES, indent=2))
            return 0
        if args.command == "validate":
            protocol = load_protocol(
                args.protocol,
                profile_override=args.profile,
                check_paths=not args.no_check_paths,
            )
            print(
                json.dumps(
                    {
                        "ok": True,
                        "suite": protocol["suite"],
                        "profile": protocol["profile"],
                        "n_cases": len(protocol["cases"]),
                        "protocol_hash": protocol["_protocol_hash"],
                    },
                    indent=2,
                )
            )
            return 0
        out = evaluate_protocol(args.protocol, args.out, profile_override=args.profile)
        print(f"evaluation complete: {out}")
        return 0
    except (ProtocolError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
