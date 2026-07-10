# tests/test_cli.py
import subprocess

from clawmarks.cli import build_parser


def test_build_is_no_longer_a_valid_subcommand():
    result = subprocess.run(
        ["python", "-m", "clawmarks.cli", "build", "scan"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr or "invalid choice" in result.stdout


def test_run_allnight_round_argument_parses():
    parser = build_parser()
    args = parser.parse_args(["run", "allnight", "--round", "2"])
    assert args.command == "run"
    assert args.round == 2


def test_serve_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["serve"])
    assert args.command == "serve"
