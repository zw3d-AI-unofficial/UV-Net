import argparse
import pathlib
import time
from datasets.fusion_joint import JointGraphDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from uvnet.models import JointPrediction
import wandb
import csv

parser = argparse.ArgumentParser("Train a joint prediction model")
parser.add_argument(
    "traintest", choices=("train", "test", "traintest"), help="Whether to train or test"
)
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension for UV-Net's embeddings")
parser.add_argument("--out_dim", type=int, default=64, help="Output dimension for SimCLR projection head")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=8,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="joint",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="Use wandb",
)
parser.add_argument(
    "--wandb_run",
    type=str,
    default="",
    help="Resume a previous wandb run",
)
parser.add_argument(
    "--random_rotate",
    action="store_true",
    default=False,
    help="Use random rotate",
)
parser.add_argument(
    "--channels",
    type=str,
    default="points,trimming_mask",
    help="Input channels to use as a string separated by commas.\
            Can include: points, normals, tangents, trimming_mask"
)
parser.add_argument(
    "--max_node_count",
    type=int,
    default=1024,
    help="Restrict training data to graph pairs with under this number of nodes.\
            Set to 0 to train on all data."
)
parser.add_argument(
    "--node_emb_dim",
    type=int,
    default=64,
    help="Restrict training data to graph pairs with under this number of nodes.\
            Set to 0 to train on all data."
)
parser.add_argument(
    "--train_label_scheme",
    type=str,
    default="Joint",
    help="Labels to use for training as a string separated by commas.\
            Can include: Joint, Ambiguous, JointEquivalent, AmbiguousEquivalent, Hole, HoleEquivalent\
            Note: 'Ambiguous' are referred to as 'Sibling' labels in the paper."
)
parser.add_argument(
    "--test_label_scheme",
    type=str,
    default="Joint,JointEquivalent",
    help="Labels to use for testing as a string separated by commas.\
            Can include: Joint, Ambiguous, JointEquivalent, AmbiguousEquivalent, Hole, HoleEquivalent\
            Note: 'Ambiguous' are referred to as 'Sibling' labels in the paper."
)
parser.add_argument(
    "--pretrained_model", 
    type=str,
    default=None,
    help="Pretrained model to use."
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="Initial learning rate."
)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_path = str(results_path.joinpath(month_day, hour_min_second))
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_last=True,
)

if args.wandb:
    if args.checkpoint is not None and args.wandb_run != "":
        wandb.init(project="joinable-pretrained", id=args.wandb_run, resume="allow")
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=WandbLogger(
            project="joinable-pretrained", 
            entity="fusiqiao101", 
            name=month_day+"_"+hour_min_second
        ),
        accelerator="dp",
        resume_from_checkpoint=args.checkpoint
    )
else:
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(
            str(results_path), name=month_day, version=hour_min_second,
        ),
        accelerator="dp",
        resume_from_checkpoint=args.checkpoint
    )

if args.traintest == "train" or args.traintest == "traintest":
    # Train/val
    seed_everything(seed=args.seed, workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
Joint prediction model training
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = JointPrediction(
        latent_dim=args.latent_dim, 
        node_emb_dim=args.node_emb_dim,
        out_dim=args.out_dim, 
        lr=args.lr, 
        channels=args.channels
    )
    train_data = JointGraphDataset(root_dir=args.dataset_path, split="train", random_rotate=args.random_rotate, max_node_count=args.max_node_count, label_scheme=args.train_label_scheme)
    val_data = JointGraphDataset(root_dir=args.dataset_path, split="val", max_node_count=16384, label_scheme=args.train_label_scheme)
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = val_data.get_dataloader(batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    trainer.fit(model, train_loader, val_loader)
if args.traintest == "test" or args.traintest == "traintest":
    # Test
    if args.traintest == "test":
        assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
        checkpoint = args.checkpoint
        checkpoint_path = '/'.join(checkpoint.split('/')[:-1])
    else:
        checkpoint = checkpoint_path + "/best.ckpt"
    model = JointPrediction.load_from_checkpoint(
        latent_dim=args.latent_dim, 
        node_emb_dim=args.node_emb_dim,
        out_dim=args.out_dim, 
        lr=args.lr, 
        channels=args.channels, 
        checkpoint_path=checkpoint
    )
    if args.gpus is not None:
        model = model.cuda()
    test_data = JointGraphDataset(root_dir=args.dataset_path, split="test", label_scheme=args.test_label_scheme)
    test_loader = test_data.get_dataloader(batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    trainer.test(model, test_dataloaders=[test_loader], verbose=False)

    # dump result
    with open(pathlib.Path(checkpoint_path) / "results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "top_1", "top_5", "loss", "true_label_num", "top_50_pairs"])
        for row in model.test_results:
            writer.writerow(row)
    

