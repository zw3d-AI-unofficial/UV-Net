import argparse
import numpy as np
import pathlib
import time
from datasets.solidletters_contrastive import SolidLettersContrastive
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from uvnet.models import Contrastive
import faiss
import wandb

parser = argparse.ArgumentParser("UV-Net self-supervision with contrastive learning")
parser.add_argument(
    "traintest", choices=("train", "test", "traintest", "search"), help="Whether to train or test"
)
parser.add_argument("--dataset", choices=("solidletters",), default="solidletters", help="Dataset to train on")
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--size_percentage", type=float, default=1, help="Percentage of data to load")
parser.add_argument("--temperature", type=float, default=0.1, help="Temperature to use in NTXentLoss")
parser.add_argument("--latent_dim", type=int, default=128, help="Latent dimension for UV-Net's embeddings")
parser.add_argument("--out_dim", type=int, default=64, help="Output dimension for SimCLR projection head")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size; larger batches are needed for SimCLR")
parser.add_argument(
    "--num_workers",
    type=int,
    default=32,
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
    default="contrastive",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed",
)
parser.add_argument(
    "--compute_index",
    action="store_true",
    default=False,
    help="Compute and save the index file for searching",
)
parser.add_argument(
    "--wandb",
    action="store_true",
    default=False,
    help="Use wandb",
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
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_last=True,
)

if args.wandb:
    if args.checkpoint is not None:
        wandb.init(project="uvnet-contrastive", id="yyv9tddu", resume="allow")
    trainer = Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        logger=WandbLogger(
            project="uvnet-contrastive", 
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

if args.dataset == "solidletters":
    Dataset = SolidLettersContrastive
else:
    raise ValueError("Unsupported dataset")

if args.traintest == "train" or args.traintest == "traintest":
    # Train/val
    seed_everything(seed=args.seed, workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
UV-Net Contrastive Learning
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = Contrastive(latent_dim=args.latent_dim, out_dim=args.out_dim, temperature=args.temperature, batch_size=args.batch_size, channels=args.channels)
    train_data = Dataset(root_dir=args.dataset_path, split="train", size_percentage=args.size_percentage, random_rotate=args.random_rotate)
    val_data = Dataset(root_dir=args.dataset_path, split="val", size_percentage=args.size_percentage, random_rotate=args.random_rotate)
    train_loader = train_data.get_dataloader(batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = val_data.get_dataloader(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    trainer.fit(model, train_loader, val_loader)
elif args.traintest == "test" or args.traintest == "traintest":
    # Test
    assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
    model = Contrastive.load_from_checkpoint(args.checkpoint)
    if args.gpus is not None:
        model = model.cuda()

    test_data = Dataset(root_dir=args.dataset_path, split="test", size_percentage=args.size_percentage)
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    test_outputs = model.get_embeddings_from_dataloader(test_loader)

    # K-means clustering on embeddings
    cluster_ami = model.clustering(test_outputs, num_clusters=test_data.num_classes(), standardize=False)
    print(f"Clustering AMI score on test set: {cluster_ami:2.3f}")
    cluster_ami = model.clustering(test_outputs, num_clusters=test_data.num_classes(), standardize=True)
    print(f"Clustering AMI score on standardized test set: {cluster_ami:2.3f}")

    # Linear SVM classification on embeddings
    train_data = Dataset(root_dir=args.dataset_path, split="train", size_percentage=args.size_percentage)
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False
    )
    train_outputs = model.get_embeddings_from_dataloader(train_loader)
    svm_acc = model.linear_svm_classification(train_outputs, test_outputs)
    print(f"Linear SVM classification accuracy (%) on test set: {svm_acc * 100.0:2.3f}")
else:
    # Search
    assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
    model = Contrastive.load_from_checkpoint(args.checkpoint)
    if args.gpus is not None:
        model = model.cuda()
    all_data = Dataset(root_dir=args.dataset_path, split="all", size_percentage=args.size_percentage)

    index_path = str(pathlib.Path(args.checkpoint).parent.joinpath("index_file.index"))
    index = None
    if args.compute_index:
        all_loader = all_data.get_dataloader(
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False
        )
        outputs = model.get_embeddings_from_dataloader(all_loader)
        embeddings = outputs["embeddings"]
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, index_path)
    else:
        index = faiss.read_index(index_path)
    
    model.eval()
    for data in all_data:
        bg = data["graph"].to(model.device)
        bg = model._permute_graph_data_channels(bg)
        _, emb = model.model(bg)
        distances, indices = index.search(emb.detach().cpu().numpy(), k=4)
        closest_parts = "\n".join([f"  {all_data[indices[0][i]]['filename']}, {distances[0][i]}" for i in range(1, 4)])
        print(f"Closest parts of {data['filename']}:\n{closest_parts}")
