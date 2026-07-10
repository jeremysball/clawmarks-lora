import argparse
import importlib

# name -> "module.path:function" for each build target. Resolved lazily per-target so
# e.g. `clawmarks build thumbnails` only imports PIL, not every build module's dependencies
# (uncanny_gallery alone pulls in torch/numpy/transformers).
_BUILD_MODULES = {
    "scan": "clawmarks.build.scan_gallery",
    "archive": "clawmarks.build.elite_archive",
    "coverage": "clawmarks.build.coverage_map",
    "map": "clawmarks.build.map_view",
    "redundancy": "clawmarks.build.redundancy_view",
    "novelty-decay": "clawmarks.build.novelty_decay",
    "lineage": "clawmarks.build.lineage_view",
    "solution-map": "clawmarks.build.solution_map",
    "similarity": "clawmarks.build.similarity_index",
    "thumbnails": "clawmarks.build.thumbnails",
    "explore-hub": "clawmarks.build.explore_hub",
    "seeds": "clawmarks.build.seed_browser",
    "probe-report": "clawmarks.build.probe_report",
    "uncanny-gallery": "clawmarks.build.uncanny_gallery",
    "preference-rank": "clawmarks.build.preference_rank",
    "rate": "clawmarks.build.rate_page",
}


def _build_target_main(name):
    return importlib.import_module(_BUILD_MODULES[name]).main


def build_parser():
    parser = argparse.ArgumentParser(prog="clawmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve")

    # Choices must be known at parse time, so enumerate the keys without importing the build
    # modules themselves (which would pull in torch/transformers). The actual callables are
    # resolved lazily in main() via _build_targets().
    build_p = sub.add_parser("build")
    build_p.add_argument("target", choices=["all", *_BUILD_MODULES])
    build_p.add_argument(
        "--use-predicted-preference", action="store_true", default=False,
        help="Stage 5b, archive target only: rank each cell's fallback by predicted preference "
             "instead of novelty. Defaults off; requires a trained preference model.",
    )

    run_p = sub.add_parser("run")
    run_sub = run_p.add_subparsers(dest="run_target", required=True)
    allnight_p = run_sub.add_parser("allnight")
    allnight_p.add_argument("--round", type=int, choices=[1, 2], required=True)
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
        return serve_main()

    if args.command == "build":
        # Only the archive target understands --use-predicted-preference; forwarding it to
        # every target would raise "unrecognized arguments" in any target that parses its own
        # argv (e.g. thumbnails).
        archive_argv = ["--use-predicted-preference"] if args.use_predicted_preference else []
        if args.target == "all":
            for name in _BUILD_MODULES:
                _build_target_main(name)(archive_argv if name == "archive" else [])
        else:
            fn_argv = archive_argv if args.target == "archive" else []
            _build_target_main(args.target)(fn_argv)
        return 0

    if args.command == "run":
        from clawmarks.search.driver import main as driver_main
        run_argv = ["--round", str(args.round)]
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
