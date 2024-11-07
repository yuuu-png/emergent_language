import argparse

import wandb

from train import main as train_main


def main(config: argparse.Namespace):
    api = wandb.Api()
    run = api.run(config.id)
    run_config = run.config
    run_config.update(vars(config))
    config = argparse.Namespace(**run.config)

    train_main(config)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, required=True)

    # Learning parameters
    parser.add_argument("--gpu", nargs="*", default=None, type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epoch", type=int, default=3000)
    parser.add_argument("--early_stopping_patience", type=int, default=100)
    parser.add_argument("--limit_batches", type=float, default=None)

    # Wandb parameters
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--project", type=str, default="simclr")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--images_per_batch", type=int, default=20)
    parser.add_argument("--batches_to_log", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    config = parser.parse_args()
    main(config)
