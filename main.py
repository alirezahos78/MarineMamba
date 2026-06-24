import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpyMamba experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset config names to run, e.g. aqua20_clip, aqua20, or fathomnet_pretrain.",
    )
    parser.add_argument(
        "--branch-settings",
        nargs="+",
        default=None,
        help="Branch settings to run, e.g. full no_local_2.",
    )
    parser.add_argument("--data-dir", help="Override the selected dataset's data directory.")
    parser.add_argument("--num-classes", type=int, help="Override the selected dataset's class count.")
    parser.add_argument("--epochs", type=int, help="Override epoch count.")
    parser.add_argument("--batch-size", type=int, help="Override batch size.")
    parser.add_argument("--pretrained-path", help="Load compatible weights before training.")
    parser.add_argument(
        "--freeze-first-blocks-on-transfer",
        type=int,
        help="Freeze this many first Mamba blocks after loading --pretrained-path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    from spiral_project.trainer import run_experiments, save_and_print_results

    overrides = {
        "data_dir": args.data_dir,
        "num_classes": args.num_classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "pretrained_path": args.pretrained_path,
        "freeze_first_blocks_on_transfer": args.freeze_first_blocks_on_transfer,
    }
    results = run_experiments(
        datasets_to_run=args.datasets,
        branch_settings=args.branch_settings,
        config_overrides=overrides,
    )
    save_and_print_results(results)


if __name__ == "__main__":
    main()
