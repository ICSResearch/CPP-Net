import argparse
import datetime
import numpy as np
import time
import json
import os
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
from timm.models import create_model
from timm.utils import ModelEma

from utils.optim_factory import create_optimizer
from utils.engine import train_one_epoch, evaluate
from utils.utils import NativeScalerWithGradNormCount as NativeScaler
from utils.utils import str2bool
import utils.utils as utils
from dataset import *
from CPP import *


def get_args_parser():
    parser = argparse.ArgumentParser("training", add_help=False)
    parser.add_argument("--batch_size", default=32, type=int, help="Per GPU batch size")
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--update_freq", default=1, type=int, help="gradient accumulation steps"
    )

    # EMA related parameters
    parser.add_argument("--model_ema", type=str2bool, default=False)
    parser.add_argument("--model_ema_decay", type=float, default=0.9999, help="")
    parser.add_argument("--model_ema_force_cpu", type=str2bool, default=False, help="")
    parser.add_argument(
        "--model_ema_eval",
        type=str2bool,
        default=False,
        help="Using ema to eval during training.",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="cpp8",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=96, type=int, help="image size")
    parser.add_argument("--cs_ratio", default=10, type=int)
    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    # Optimization parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=2e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 16",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )

    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=-1,
        metavar="N",
        help="num of steps to warmup LR, will overload warmup_epochs if set > 0",
    )
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt_eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt_betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=None,
        help="""Final value of the weight decay.""",
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path", default="../data/train", type=str, help="dataset path"
    )
    parser.add_argument(
        "--output_dir",
        default="./model",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./log", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--eval_data_path",
        default="../data/val",
        type=str,
        help="dataset path for evaluation",
    )

    parser.add_argument("--auto_resume", type=str2bool, default=True)
    parser.add_argument("--save_ckpt", type=str2bool, default=True)
    parser.add_argument("--save_ckpt_freq", default=5, type=int)
    parser.add_argument("--save_ckpt_num", default=10, type=int)

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--eval", type=str2bool, default=False, help="Perform evaluation only"
    )
    parser.add_argument(
        "--dist_eval",
        type=str2bool,
        default=True,
        help="Enabling distributed evaluation",
    )
    parser.add_argument(
        "--disable_eval",
        type=str2bool,
        default=False,
        help="Disabling evaluation during training",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        type=str2bool,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    # Evaluation parameters
    parser.add_argument("--crop_pct", type=float, default=None)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", type=str2bool, default=False)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--use_amp",
        type=str2bool,
        default=False,
        help="Use apex AMP (Automatic Mixed Precision) or not",
    )
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print(args)
    device = torch.device(args.device)
    # torch.autograd.set_detect_anomaly(True)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    if args.disable_eval:
        args.dist_eval = False
        dataset_val = None
    else:
        dataset_val = build_dataset(is_train=False, args=args)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train,
        num_replicas=num_tasks,
        rank=global_rank,
        shuffle=True,
        seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                "This will slightly alter validation results as extra duplicate entries are added to achieve "
                "equal num of samples per-process."
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
    else:
        data_loader_val = None

    mixup_fn = None

    model = create_model(args.model, ratio=args.cs_ratio).to(device)
    # model_path = f"./model/checkpoint-{args.model}-25-best.pth"
    # checkpoint = torch.load(model_path, map_location="cpu")
    # checkpoint['model'].pop('A')
    # model.load_state_dict(checkpoint["model"], strict=False)
    # print(f"load {model_path}")

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print("number of params:", n_parameters)

    eff_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // eff_batch_size

    if args.lr is None:
        args.lr = args.blr * eff_batch_size / args.batch_size

    print("base lr: %.2e" % (args.lr * args.batch_size / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.update_freq)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=False
        )
        model_without_ddp = model.module

    optimizer = create_optimizer(args, model_without_ddp, skip_list=None)
    loss_scaler = NativeScaler()

    criterion = torch.nn.MSELoss()
    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args,
        model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        model_name=args.model,
        ratio=args.cs_ratio,
        model_ema=model_ema,
    )

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%"
        )
        return

    max_psnr = 0.0
    if args.model_ema and args.model_ema_eval:
        max_psnr_ema = 0.0

    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                    model_name=args.model,
                    ratio=args.cs_ratio,
                    model_ema=model_ema,
                )
        if data_loader_val is not None:
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
            print(
                f"PSNR of the model on the {len(dataset_val)} test images: {test_stats['psnr']:.4f}"
            )
            if max_psnr < test_stats["psnr"]:
                max_psnr = test_stats["psnr"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_name=args.model,
                        ratio=args.cs_ratio,
                        model_ema=model_ema,
                    )
            print(f"Max PSNR: {max_psnr:.4f}")

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats["psnr"], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats["loss"], head="perf", step=epoch)

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(
                    data_loader_val, model_ema.ema, device, use_amp=args.use_amp
                )
                print(
                    f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['psnr']:.4f}%"
                )
                if max_psnr_ema < test_stats_ema["psnr"]:
                    max_psnr_ema = test_stats_ema["psnr"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args,
                            model=model,
                            model_without_ddp=model_without_ddp,
                            optimizer=optimizer,
                            loss_scaler=loss_scaler,
                            epoch="best-ema",
                            model_name=args.model,
                            ratio=args.cs_ratio,
                            model_ema=model_ema,
                        )
                    print(f"Max EMA PSNR: {max_psnr_ema:.4f}%")
                if log_writer is not None:
                    log_writer.update(
                        test_acc1_ema=test_stats_ema["psnr"], head="perf", step=epoch
                    )
                log_stats.update(
                    {**{f"test_{k}_ema": v for k, v in test_stats_ema.items()}}
                )
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, f"{args.model}_{args.cs_ratio}_log.txt"),
                mode="a",
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
