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
parser.add_argument(
    "--dataset",
    type=str,
    default="/home/share/brep/zw3d/zw3d-joinable-dataset",
    help="Dataset path."
)
parser.add_argument(
    "--latent_dim", 
    type=int, 
    default=512, 
    help="Latent dimension for UV-Net's embeddings"
)
parser.add_argument(
    "--batch_size", 
    type=int, 
    default=16, 
    help="Batch size"
)
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
    "--input_features",
    type=str,
    default="type,area,length,points,normals,tangents,trimming_mask",
    help="Input features to use as a string separated by commas.\
            Can include: points, normals, tangents, trimming_mask,\
            axis_pos, axis_dir, bounding_box, type, parameter\
            area, circumference\
            length, start_point, middle_point, end_point"
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
    default=0.0001,
    help="Initial learning rate."
)
parser.add_argument(
    "--test_split",
    type=str,
    default="test",
    choices=("train", "val", "test"),
    help="Test split to use during evaluation."
)
parser.add_argument(
    "--n_head",
    type=int,
    default=8,
    help="Number of attention heads."
)
parser.add_argument(
    "--n_layer_gat",
    type=int,
    default=2,
    help="Number of Graph-Attention layers."
)
parser.add_argument(
    "--n_layer_sat",
    type=int,
    default=2,
    help="Number of Self-Attention layers."
)
parser.add_argument(
    "--n_layer_cat",
    type=int,
    default=2,
    help="Number of Cross-Attention layers."
)
parser.add_argument(
    "--bias",
    action="store_true",
    default=False,
    help="Use bias in mlp."
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.2,
    help="Dropout rate."
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
        wandb.init(project="joinable_binary", id=args.wandb_run, resume="allow")
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=WandbLogger(
            project="joinable_binary", 
            entity="fusiqiao101", 
            name=month_day+"_"+hour_min_second
        ),
        resume_from_checkpoint=args.checkpoint
    )
else:
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=TensorBoardLogger(
            str(results_path), name=month_day, version=hour_min_second,
        ),
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
        input_features=args.input_features,
        emb_dim=args.latent_dim,
        n_head=args.n_head,
        n_layer_gat=args.n_layer_gat,
        n_layer_sat=args.n_layer_sat,
        n_layer_cat=args.n_layer_cat,
        bias=args.bias,
        dropout=args.dropout,
        lr=args.lr
    )
    train_data = JointGraphDataset(root_dir=args.dataset, split="train", random_rotate=args.random_rotate, max_node_count=2048)
    val_data = JointGraphDataset(root_dir=args.dataset, split="val", max_node_count=16384)
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = val_data.get_dataloader(batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    trainer.fit(model, train_loader, val_loader)
if args.traintest == "test" or args.traintest == "traintest":
    # Test
    if args.traintest == "test":
        assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
        checkpoint = args.checkpoint
    else:
        checkpoint = checkpoint_path + "/best.ckpt"
    model = JointPrediction.load_from_checkpoint(
        input_features=args.input_features,
        emb_dim=args.latent_dim,
        n_head=args.n_head,
        n_layer_gat=args.n_layer_gat,
        n_layer_sat=args.n_layer_sat,
        n_layer_cat=args.n_layer_cat,
        bias=args.bias,
        dropout=args.dropout,
        lr=args.lr,
        checkpoint_path=checkpoint
    )
    if args.gpus is not None:
        model = model.cuda()
    test_data = JointGraphDataset(root_dir=args.dataset, split=args.test_split)
    test_loader = test_data.get_dataloader(batch_size=1, shuffle=False, num_workers=args.num_workers, drop_last=False)
    trainer.test(model, test_dataloaders=[test_loader])

    # dump result
    with open(pathlib.Path(checkpoint_path) / "results.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "loss", "pred", "label"])
        for row in model.test_results:
            writer.writerow(row)
    

