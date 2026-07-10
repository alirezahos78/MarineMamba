import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def parse_args():
    parser = argparse.ArgumentParser(description="Train MarineMamba.")
    parser.add_argument(
        "--configs", nargs="+",
        default=["aqua20_dual_hybrid_128_focal_balanced"],
        help="Config names to run.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=SEEDS,
        help="Random seeds (default: 0-9).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from marinemamba.trainer import run_multi_seed
    run_multi_seed(config_names=args.configs, seeds=args.seeds)


if __name__ == "__main__":
    main()
