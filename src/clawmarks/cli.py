import argparse


def _build_targets():
    # Imported lazily so the parser-only path (build_parser, used by tests) does not require
    # the build modules' heavy imports (torch, transformers, PIL) to be installed.
    from clawmarks.build import (
        scan_gallery, elite_archive, coverage_map, map_view, redundancy_view,
        novelty_decay, lineage_view, solution_map, similarity_index, thumbnails,
        explore_hub, seed_browser, probe_report, uncanny_gallery, rate_page, preference_rank,
    )
    return {
        "scan": scan_gallery.main,
        "archive": elite_archive.main,
        "preference-rank": preference_rank.main,
        "coverage": coverage_map.main,
        "map": map_view.main,
        "redundancy": redundancy_view.main,
        "novelty-decay": novelty_decay.main,
        "lineage": lineage_view.main,
        "solution-map": solution_map.main,
        "similarity": similarity_index.main,
        "thumbnails": thumbnails.main,
        "explore-hub": explore_hub.main,
        "seeds": seed_browser.main,
        "probe-report": probe_report.main,
        "uncanny-gallery": uncanny_gallery.main,
        "rate": rate_page.main,
    }


def build_parser():
    parser = argparse.ArgumentParser(prog="clawmarks")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("serve")

    # Choices must be known at parse time, so enumerate the keys without importing the build
    # modules themselves (which would pull in torch/transformers). The actual callables are
    # resolved lazily in main() via _build_targets().
    build_p = sub.add_parser("build")
    build_p.add_argument(
        "target",
        choices=["all", "scan", "archive", "coverage", "map", "redundancy", "novelty-decay",
                 "lineage", "solution-map", "similarity", "thumbnails", "explore-hub", "seeds",
                 "probe-report", "uncanny-gallery", "rate", "preference-rank"],
    )
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
    probe_sub.add_parser("train")

    pod_p = sub.add_parser("pod")
    pod_sub = pod_p.add_subparsers(dest="pod_action", required=True)
    for action in ("bring-up", "pause", "terminate", "ssh", "get", "put"):
        pod_sub.add_parser(action)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "serve":
        from clawmarks.curation_server import main as serve_main
        return serve_main()

    if args.command == "build":
        targets = _build_targets()
        # Only the archive target understands --use-predicted-preference; forwarding it to
        # every target would raise "unrecognized arguments" in any target that parses its own
        # argv (e.g. thumbnails).
        archive_argv = ["--use-predicted-preference"] if args.use_predicted_preference else []
        if args.target == "all":
            for name, fn in targets.items():
                fn(archive_argv if name == "archive" else [])
        else:
            fn_argv = archive_argv if args.target == "archive" else []
            targets[args.target](fn_argv)
        return 0

    if args.command == "run":
        from clawmarks.search.driver import main as driver_main
        run_argv = ["--round", str(args.round)]
        if args.use_predicted_preference:
            run_argv.append("--use-predicted-preference")
        return driver_main(run_argv)

    if args.command == "probe" and args.probe_target == "train":
        from clawmarks.probe.train import main as train_main
        return train_main()

    if args.command == "pod":
        from clawmarks.compute import runpod
        action_map = {
            "bring-up": lambda: runpod.bring_up(gpu_priority=["RTX 4090", "RTX 3090", "RTX A5000"]),
            "pause": lambda: runpod.pause(input("pod id: ")),
            "terminate": lambda: runpod.terminate(input("pod id: ")),
        }
        if args.pod_action in action_map:
            action_map[args.pod_action]()
        return 0

    parser.error(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
