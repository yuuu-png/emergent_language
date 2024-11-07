import argparse
import wandb

import train

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", nargs="*", default=None, type=int)
parser.add_argument("--project", type=str, default="pretrain-sweep")
parser.add_argument("--fast_dev_run", action="store_true")
parser.add_argument("--count", type=int, default=1000)
parser.add_argument("--sweep_id", type=str)
parser.add_argument("--num_workers", type=int, default=None)
parser.add_argument("--epoch", type=int, default=None)
parser.add_argument("--limit_batches", type=float, default=None)
args = parser.parse_args()

# Define sweep config
sweep_configuration = {
    "method": "bayes",
    "name": "gumbel softmax with standard backpropagation",
    "metric": {"goal": "minimize", "name": "val_reconstruction_loss"},
    "parameters": {
        "lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        "batch_size": {"values": [64, 128, 256, 512]},
        "hidden_dim": {"values": [32, 64, 128, 256]},
    },
}


def sweep_func():
    run = wandb.init(project=args.project)
    config = run.config

    config.gpu = args.gpu
    config.seed = 0
    # config.batch_size = 512
    config.num_workers = args.num_workers
    config.epoch = 100 if args.epoch is None else args.epoch  # changed
    # config.lr = 1e-4
    config.early_stopping_patience = 15  # changed
    config.data_augmentation = False
    config.limit_batches = None if args.limit_batches is None else args.limit_batches

    config.dataset = "mnist"
    # config.hidden_dim = 32
    config.backbone = "monotone28"
    config.backbone_checkpoint = None

    config.offline = False
    config.project = args.project
    config.name = None
    config.fast_dev_run = args.fast_dev_run
    config.images_per_batch = 50
    config.batches_to_log = 10  # changed
    config.save_interval = 10

    train.main(config=config, sweep=True)
    run.finish()


sweep_id = args.sweep_id
if sweep_id is None:
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)

wandb.agent(sweep_id, function=sweep_func, count=args.count, project=args.project)
