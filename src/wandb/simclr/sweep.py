import argparse
import wandb

import train

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", nargs="*", default=None, type=int)
parser.add_argument("--project", type=str, default="simclr-sweep")
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
    "name": "cifar10 batch size 512",
    # in some cases, this metric might be "val_simclr_loss"
    "metric": {"goal": "minimize", "name": "val_standard_loss"},
    "parameters": {
        "speaker_lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        "listener_lr": {"distribution": "log_uniform_values", "min": 1e-6, "max": 1e-2},
        # "entropy_coeff": {
        #     "distribution": "log_uniform_values",
        #     "min": 1e-6,
        #     "max": 1e-2,
        # },
        "cosine_temperature": {
            "distribution": "log_uniform_values",
            "min": 0.001,
            "max": 1,
        },
    },
}


def sweep_func():
    run = wandb.init(project=args.project)
    config = run.config

    config.gpu = args.gpu
    config.seed = 0
    config.batch_size = 512  # changed
    config.num_workers = args.num_workers
    config.epoch = 100 if args.epoch is None else args.epoch  # changed
    config.backbone_lr = 1e-4
    # config.speaker_lr = 5e-4
    # config.listener_lr = 1e-4
    config.early_stopping_patience = 15  # changed
    config.wo_freeze = False
    config.data_augmentation = True  # changed
    config.augmentation_min_scale = 0.2  # changed
    config.limit_batches = None if args.limit_batches is None else args.limit_batches

    config.dataset = "cifar10"  # changed
    config.vocab_size = 32  # changed
    config.max_len = 8  # changed
    config.length_cost = 0
    config.similarity = "cosine"
    # config.cosine_temperature = 0.1

    config.backbone = "pretrained_vit_b_16"  # changed
    config.backbone_checkpoint = None
    config.arch = "transformer"
    config.hidden_dim = 256  # changed
    config.embed_dim = 10
    config.nhead = 4
    config.num_layers = 3
    config.dropout = 0.1
    config.lazy_speaker = False
    config.lazy_speaker_beta1 = 45
    config.lazy_speaker_beta2 = 10
    config.classifier = False
    config.classifier_lr = 1e-3
    config.classifier_hidden_dim = 64
    config.cls_loss_coeff = 1

    config.gumbel_softmax = True  # changed
    config.straight_through = False
    config.detach_message = False
    config.wo_policy_loss = True  # changed
    config.entropy_coeff = 0  # changed

    config.offline = False
    config.project = args.project
    config.name = None
    config.fast_dev_run = args.fast_dev_run
    config.images_per_batch = 20  # changed
    config.batches_to_log = 1
    config.save_interval = 99  # changed

    train.main(config=config, sweep=True)
    run.finish()


sweep_id = args.sweep_id
if sweep_id is None:
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project)

wandb.agent(sweep_id, function=sweep_func, count=args.count, project=args.project)
