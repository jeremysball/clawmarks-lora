import argparse


def build_parser():
    parser = argparse.ArgumentParser(prog="clawmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve")

    run_p = sub.add_parser("run")
    run_sub = run_p.add_subparsers(dest="run_target", required=True)
    allnight_p = run_sub.add_parser("allnight")
    allnight_p.add_argument("--expedition", required=True)
    allnight_p.add_argument("--leg", required=True)
    allnight_p.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b: build the exploit pool from the trained preference model's top picks "
             "instead of yes-rated images. Defaults off; requires a trained preference model.",
    )
    allnight_p.set_defaults(command="run")

    probe_p = sub.add_parser("probe")
    probe_sub = probe_p.add_subparsers(dest="probe_target", required=True)
    # No args declared here: probe/train.py's own argparse (--name, --max-train-steps, etc.)
    # owns validation. Trailing args are forwarded via parse_known_args in main().
    probe_sub.add_parser("train")

    pod_p = sub.add_parser("pod")
    pod_sub = pod_p.add_subparsers(dest="pod_action", required=True)
    for action in ("bring-up", "pause", "terminate", "ssh", "get", "put"):
        pod_sub.add_parser(action)

    return parser


def main(argv=None):
    parser = build_parser()
    # parse_known_args, not parse_args: `probe train` forwards its unrecognized trailing args
    # (--name, --max-train-steps, ...) to probe/train.py's own parser rather than declaring
    # them twice. Every other command still gets strict validation below.
    args, extra = parser.parse_known_args(argv)

    if args.command == "probe" and args.probe_target == "train":
        from clawmarks.probe.train import main as train_main
        return train_main(extra)

    if extra:
        parser.error(f"unrecognized arguments: {' '.join(extra)}")

    if args.command == "serve":
        from clawmarks.curation_server import main as serve_main
        return serve_main([])

    if args.command == "run":
        from clawmarks.search.driver import main as driver_main
        run_argv = ["--expedition", args.expedition, "--leg", args.leg]
        if args.use_predicted_preference:
            run_argv.append("--use-predicted-preference")
        return driver_main(run_argv)

    if args.command == "pod":
        from clawmarks.compute import runpod
        action_map = {
            "bring-up": lambda: runpod.bring_up(gpu_priority=["RTX 4090", "RTX 3090", "RTX A5000"]),
            "pause": lambda: runpod.pause(input("pod id: ")),
            "terminate": lambda: runpod.terminate(input("pod id: ")),
            "ssh": lambda: runpod.ssh(input("pod id: "), input("command: ")),
            "get": lambda: runpod.get(input("pod id: "), input("remote path: "), input("local path: ")),
            "put": lambda: runpod.put(input("pod id: "), input("local path: "), input("remote path: ")),
        }
        action_map[args.pod_action]()
        return 0

    parser.error(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
