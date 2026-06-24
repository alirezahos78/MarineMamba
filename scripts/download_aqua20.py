import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from spiral_project.config import get_dataset_config  # noqa: E402
from spiral_project.data import prepare_aqua20  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Download AQUA20 and create ImageFolder-compatible train/test directories."
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Destination directory (default: <project>/data/aqua20).",
    )
    args = parser.parse_args()

    config = get_dataset_config("aqua20")
    output_dir = os.path.abspath(args.output_dir or config["data_dir"])
    train_dir, test_dir = prepare_aqua20(output_dir)
    print(f"✅ AQUA20 train split: {train_dir}")
    print(f"✅ AQUA20 test split:  {test_dir}")


if __name__ == "__main__":
    main()
