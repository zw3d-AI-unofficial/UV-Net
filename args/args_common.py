"""

Common args across different training scripts

"""

import argparse
import torch


def get_parser():
    """Return the shared parser"""
    parser = argparse.ArgumentParser("Train a joint prediction model")
    parser.add_argument(
        "--traintest",
        type=str,
        default="traintest",
        choices=("train", "test", "traintest", "randomtest"),
        help="Whether to train or test."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/zw3d-joinable-dataset",
        help="Dataset path."
    )
    parser.add_argument(
        "--delete_cache",
        action="store_true",
        default=False,
        help="Delete the dataset cache."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of samples in a batch."
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        choices=("train", "val", "test", "mix_test"),
        help="Test split to use during evaluation."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last",
        help="Checkpoint to load for evaluation, e.g. last, best, etc..."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume checkpoint"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of data samples."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="experiment",
        help="Experiment name to use for file naming."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="-1",
        help="Which gpu device to use, use -1 for all gpus"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="ddp",
        choices=("dp", "ddp", "ddp_spawn", "ddp_sharded", "ddp_sharded_spawn", "None"),
        help="Accelerator to use for multi-gpu training."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of threads to uses for data loading."
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers to use in the torch dataloader."
    )
    parser.add_argument(
        "--random_rotate",
        action="store_true",
        default=False,
        help="Randomly rotate the training set."
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="results_onnx",
        help="Directory to save the best and last checkpoints and logs."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed to use"
    )
    parser.add_argument(
        "--skip_far",
        action="store_true",
        default=False,
        help="Skip pairs that are far away."
    )
    parser.add_argument(
        "--skip_interference",
        action="store_true",
        default=False,
        help="Skip pairs that have interference."
    )
    parser.add_argument(
        "--skip_nurbs",
        action="store_true",
        default=False,
        help="Skip true label with nurbs curve or surface as entity."
    )
    parser.add_argument(
        "--skip_synthetic",
        action="store_true",
        default=False,
        help="Skip synthetic joints."
    )
    parser.add_argument(
        "--joint_type",
        type=str,
        default="all",
        help="Joint type to use as a string separated by commas.\
                Can include: Other, Coincident, Tangent, Concentric,\
                Parallel, Perpendicular"
    )
    return parser


def get_args(parser):
    """Get the input args"""
    args = parser.parse_args()
    # If we don't have cuda available, use CPU
    if not torch.cuda.is_available():
        args.gpus = None
    return args
