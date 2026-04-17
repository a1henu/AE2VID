import argparse
import sys


def _run(module_name, config_path):
    sys.argv = [f"{module_name}.py", "--config", config_path]
    module = __import__(module_name)
    module.main()


def main():
    parser = argparse.ArgumentParser(description="AE2VID training entrypoint")
    subparsers = parser.add_subparsers(dest="stage", required=True)

    adapter = subparsers.add_parser(
        "adapter",
        help="Stage 1: train the aperture/HSG adapter initialized from V2V-E2VID.",
    )
    adapter.add_argument("--config", required=True, help="Path to the stage-1 YAML config.")

    ae2vid = subparsers.add_parser(
        "ae2vid",
        help="Stage 2: train the full AE2VID model.",
    )
    ae2vid.add_argument("--config", required=True, help="Path to the stage-2 YAML config.")

    args = parser.parse_args()
    if args.stage == "adapter":
        _run("train_adapter", args.config)
    elif args.stage == "ae2vid":
        _run("train_ae2vid", args.config)
    else:
        raise ValueError(f"Unknown training stage: {args.stage}")


if __name__ == "__main__":
    main()
