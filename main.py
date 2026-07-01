import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpyMamba (PyramidCLIPSpyMamba).")
    parser.add_argument("--configs", nargs="+", default=["aqua20_pyramid_hybrid_128_focal_balanced"],
                        help="Config names to run.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    from spymamba.trainer import run_experiments, save_and_print_results
    results = run_experiments(config_names=args.configs, seed=args.seed)
    save_and_print_results(results)


if __name__ == "__main__":
    main()
