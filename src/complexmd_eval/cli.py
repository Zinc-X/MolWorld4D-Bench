"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .codex_integration import (
    doctor_report,
    install_skill,
    render_doctor,
    render_json,
    skill_status,
)
from .profiles import PROFILES


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
    doctor = sub.add_parser("doctor", help="check runtime, dependencies, protocol, and Codex skill")
    doctor.add_argument("--profile", choices=sorted(PROFILES))
    doctor.add_argument("--protocol", type=Path)
    doctor.add_argument("--no-check-paths", action="store_true")
    doctor.add_argument("--strict", action="store_true", help="fail when optional profile features are unavailable")
    doctor.add_argument("--json", action="store_true", dest="json_output")
    skill = sub.add_parser("skill", help="inspect or install the bundled Codex skill")
    skill_sub = skill.add_subparsers(dest="skill_command", required=True)
    skill_status_parser = skill_sub.add_parser("status", help="show bundled, user, and repository skill status")
    skill_status_parser.add_argument("--repo", type=Path)
    skill_status_parser.add_argument("--json", action="store_true", dest="json_output")
    skill_install = skill_sub.add_parser("install", help="install without overwriting an existing skill")
    skill_install.add_argument("--scope", choices=("user", "repo"), default="user")
    skill_install.add_argument("--repo", type=Path)
    skill_install.add_argument("--mode", choices=("copy", "symlink"), default="copy")
    skill_install.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "profiles":
            print(json.dumps(PROFILES, indent=2))
            return 0
        if args.command == "doctor":
            report = doctor_report(
                profile=args.profile,
                protocol_path=args.protocol,
                check_paths=not args.no_check_paths,
            )
            print(render_json(report) if args.json_output else render_doctor(report))
            if not report["ok"]:
                return 1
            return 2 if args.strict and not report["complete"] else 0
        if args.command == "skill":
            if args.skill_command == "status":
                status = skill_status(args.repo)
                if args.json_output:
                    print(render_json(status))
                else:
                    print(f"bundled source: {status['source'] or 'unavailable'}")
                    for scope, item in status["targets"].items():
                        if not item.get("available"):
                            print(f"{scope}: unavailable ({item['reason']})")
                        else:
                            state = "current" if item["matches_bundled"] else "not installed or different"
                            print(f"{scope}: {state} ({item['path']})")
                return 0 if status["source_complete"] else 1
            result = install_skill(
                args.scope,
                repo_root=args.repo,
                mode=args.mode,
                dry_run=args.dry_run,
            )
            print(render_json(result))
            return 0
        if args.command == "validate":
            from .protocol import load_protocol

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
        from .pipeline import evaluate_protocol

        out = evaluate_protocol(args.protocol, args.out, profile_override=args.profile)
        print(f"evaluation complete: {out}")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
