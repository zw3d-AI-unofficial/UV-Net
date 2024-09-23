import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import (
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

import uvnet.encoders
import metrics
from uvnet.joinable import JoinABLe


class _NonLinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        """
        A 3-layer MLP with linear outputs

        Args:
            input_dim (int): Dimension of the input tensor
            num_classes (int): Dimension of the output logits
            dropout (float, optional): Dropout used after each linear layer. Defaults to 0.3.
        """
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 512, bias=False)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=dropout)
        self.linear3 = nn.Linear(256, num_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, inp):
        """
        Forward pass

        Args:
            inp (torch.tensor): Inputs features to be mapped to logits
                                (batch_size x input_dim)

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        x = F.relu(self.bn1(self.linear1(inp)))
        x = self.dp1(x)
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)
        return x


###############################################################################
# Classification model
###############################################################################


class UVNetClassifier(nn.Module):
    """
    UV-Net solid classification model
    """

    def __init__(
            self,
            num_classes,
            crv_emb_dim=64,
            srf_emb_dim=64,
            graph_emb_dim=128,
            dropout=0.3,
    ):
        """
        Initialize the UV-Net solid classification model

        Args:
            num_classes (int): Number of classes to output
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(
            in_channels=6, output_dims=crv_emb_dim
        )
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        self.clf = _NonLinearClassifier(graph_emb_dim, num_classes, dropout)

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (batch_size x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        # Per-face embeddings are ignored during solid classification
        _, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Map to logits
        out = self.clf(graph_emb)
        return out


class Classification(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """

    def __init__(self, num_classes):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetClassifier(num_classes=num_classes)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("train_acc", self.train_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("val_acc", self.val_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        labels = batch["label"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.log("test_acc", self.test_acc(preds, labels), on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


###############################################################################
# Segmentation model
###############################################################################


class UVNetSegmenter(nn.Module):
    """
    UV-Net solid face segmentation model
    """

    def __init__(
            self,
            num_classes,
            crv_in_channels=6,
            crv_emb_dim=64,
            srf_emb_dim=64,
            graph_emb_dim=128,
            dropout=0.3,
    ):
        """
        Initialize the UV-Net solid face segmentation model

        Args:
            num_classes (int): Number of classes to output per-face
            crv_in_channels (int, optional): Number of input channels for the 1D edge UV-grids
            crv_emb_dim (int, optional): Embedding dimension for the 1D edge UV-grids. Defaults to 64.
            srf_emb_dim (int, optional): Embedding dimension for the 2D face UV-grids. Defaults to 64.
            graph_emb_dim (int, optional): Embedding dimension for the graph. Defaults to 128.
            dropout (float, optional): Dropout for the final non-linear classifier. Defaults to 0.3.
        """
        super().__init__()
        # A 1D convolutional network to encode B-rep edge geometry represented as 1D UV-grids
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(
            in_channels=crv_in_channels, output_dims=crv_emb_dim
        )
        # A 2D convolutional network to encode B-rep face geometry represented as 2D UV-grids
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(
            in_channels=7, output_dims=srf_emb_dim
        )
        # A graph neural network that message passes face and edge features
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim,
        )
        # A non-linear classifier that maps face embeddings to face logits
        self.seg = _NonLinearClassifier(
            graph_emb_dim + srf_emb_dim, num_classes, dropout=dropout
        )

    def forward(self, batched_graph):
        """
        Forward pass

        Args:
            batched_graph (dgl.Graph): A batched DGL graph containing the face 2D UV-grids in node features
                                       (ndata['x']) and 1D edge UV-grids in the edge features (edata['x']).

        Returns:
            torch.tensor: Logits (total_nodes_in_batch x num_classes)
        """
        # Input features
        input_crv_feat = batched_graph.edata["x"]
        input_srf_feat = batched_graph.ndata["x"]
        # Compute hidden edge and face features
        hidden_crv_feat = self.curv_encoder(input_crv_feat)
        hidden_srf_feat = self.surf_encoder(input_srf_feat)
        # Message pass and compute per-face(node) and global embeddings
        node_emb, graph_emb = self.graph_encoder(
            batched_graph, hidden_srf_feat, hidden_crv_feat
        )
        # Repeat the global graph embedding so that it can be
        # concatenated to the per-node embeddings
        num_nodes_per_graph = batched_graph.batch_num_nodes().to(graph_emb.device)
        graph_emb = graph_emb.repeat_interleave(num_nodes_per_graph, dim=0).to(graph_emb.device)
        local_global_feat = torch.cat((node_emb, graph_emb), dim=1)
        # Map to logits
        out = self.seg(local_global_feat)
        return out


class Segmentation(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the segmenter (per-face classifier).
    """

    def __init__(self, num_classes, crv_in_channels=6):
        """
        Args:
            num_classes (int): Number of per-face classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = UVNetSegmenter(num_classes, crv_in_channels=crv_in_channels)
        # Setting compute_on_step = False to compute "part IoU"
        # This is because we want to compute the IoU on the entire dataset
        # at the end to account for rare classes, rather than within each batch
        self.train_iou = torchmetrics.IoU(
            num_classes=num_classes, compute_on_step=False
        )
        self.val_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)
        self.test_iou = torchmetrics.IoU(num_classes=num_classes, compute_on_step=False)

        self.train_accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, compute_on_step=False
        )
        self.val_accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, compute_on_step=False
        )
        self.test_accuracy = torchmetrics.Accuracy(
            num_classes=num_classes, compute_on_step=False
        )

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.train_iou(preds, labels)
        self.train_accuracy(preds, labels)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_iou", self.train_iou.compute())
        self.log("train_accuracy", self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.val_iou(preds, labels)
        self.val_accuracy(preds, labels)
        return loss

    def validation_epoch_end(self, outs):
        self.log("val_iou", self.val_iou.compute())
        self.log("val_accuracy", self.val_accuracy.compute())

    def test_step(self, batch, batch_idx):
        inputs = batch["graph"].to(self.device)
        inputs.ndata["x"] = inputs.ndata["x"].permute(0, 3, 1, 2)
        inputs.edata["x"] = inputs.edata["x"].permute(0, 2, 1)
        labels = inputs.ndata["y"]
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        preds = F.softmax(logits, dim=-1)
        self.test_iou(preds, labels)
        self.test_accuracy(preds, labels)

    def test_epoch_end(self, outs):
        self.log("test_iou", self.test_iou.compute())
        self.log("test_accuracy", self.test_accuracy.compute())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


###############################################################################
# Self-supervised model
###############################################################################

def mask_correlated_samples(batch_size, device=torch.device("cpu")):
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=bool, device=device)
    mask = mask.fill_diagonal_(0)
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


class NTXentLoss(pl.LightningModule):
    def __init__(self, temperature=0.5, batch_size=256):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        mask = mask_correlated_samples(batch_size, self.device)
        self.register_buffer("mask", mask)

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples
        within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        mask = self.mask  # self.mask_correlated_samples(batch_size)

        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N, device=positive_samples.device, dtype=torch.long)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class UVNetContrastiveLearner(nn.Module):
    def __init__(
            self,
            latent_dim,
            crv_emb_dim=64,
            srf_emb_dim=64,
            graph_emb_dim=128,
            dropout=0.3,
            out_dim=128,
            channels=["points", "trimming_mask"]
    ):
        """
        UVNetContrastivelearner
        """
        super().__init__()
        self.crv_in_channels = []
        self.srf_in_channels = []
        if "points" in channels:
            self.crv_in_channels.extend([0, 1, 2])
            self.srf_in_channels.extend([0, 1, 2])
        if "normals" in channels:
            self.srf_in_channels.extend([3, 4, 5])
        if "tangents" in channels:
            self.crv_in_channels.extend([3, 4, 5])
        self.srf_in_channels.append(6)
        self.curv_encoder = uvnet.encoders.UVNetCurveEncoder(
            in_channels=len(self.crv_in_channels), output_dims=crv_emb_dim
        )
        self.surf_encoder = uvnet.encoders.UVNetSurfaceEncoder(
            in_channels=len(self.srf_in_channels), output_dims=srf_emb_dim
        )
        self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(
            srf_emb_dim, crv_emb_dim, graph_emb_dim, hidden_dim=srf_emb_dim
        )
        # self.fc_layers = nn.Sequential(
        #     nn.Linear(graph_emb_dim, latent_dim, bias=False),
        #     nn.BatchNorm1d(latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim),
        # )
        self.projection_layer = nn.Sequential(
            nn.Linear(graph_emb_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim, bias=False),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, out_dim, bias=False),
        )

    def forward(self, bg):
        # We only use point coordinates & mask in contrastive experiments
        # TODO(pradeep): expose a better way for the user to select these channels
        nfeat = bg.ndata["x"][:, self.srf_in_channels, :, :]  # XYZ+mask channels
        efeat = bg.edata["x"][:, self.crv_in_channels, :]
        crv_feat = self.curv_encoder(efeat)
        srf_feat = self.surf_encoder(nfeat)
        node_emb, graph_emb = self.graph_encoder(bg, srf_feat, crv_feat)
        global_emb = graph_emb  # self.fc_layers(graph_emb)
        projection_out = self.projection_layer(global_emb)
        projection_out = F.normalize(projection_out, p=2, dim=-1)

        return projection_out, global_emb, node_emb


class Contrastive(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the contrastive learning model.
    """

    def __init__(
            self,
            latent_dim=128,
            crv_emb_dim=64,
            srf_emb_dim=64,
            graph_emb_dim=128,
            out_dim=128,
            temperature=0.1,
            batch_size=256,
            channels="points,trimming_mask"
    ):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = NTXentLoss(temperature=temperature, batch_size=batch_size)
        self.model = UVNetContrastiveLearner(
            latent_dim=latent_dim,
            crv_emb_dim=crv_emb_dim,
            srf_emb_dim=srf_emb_dim,
            graph_emb_dim=graph_emb_dim,
            out_dim=out_dim,
            channels=channels.split(',')
        )

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def step(self, batch, batch_idx):
        device = next(self.model.buffers()).device
        graph1, graph2 = batch["graph"].to(device), batch["graph2"].to(device)
        graph1 = self._permute_graph_data_channels(graph1)
        graph2 = self._permute_graph_data_channels(graph2)
        proj1, _ = self.model(graph1)
        proj2, _ = self.model(graph2)
        return self.loss_fn(proj1, proj2)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        loss = self.step(batch, batch_idx)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if batch is None:
            return None
        loss = self.step(batch, batch_idx)
        if loss > 5:
            print("valid loss:", loss, ", file:", batch["filename"])
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), weight_decay=1e-4)
        return optimizer

    @torch.no_grad()
    def clustering(self, data, num_clusters=26, n_init=100, standardize=False):
        if standardize:
            scaler = StandardScaler().fit(data["embeddings"])
            embeddings = scaler.transform(data["embeddings"].copy())
        else:
            embeddings = data["embeddings"]
        kmeans = KMeans(init="random", n_clusters=num_clusters, n_init=n_init, max_iter=100000)
        print(f"Fitting K-Means with {num_clusters} clusters...")
        kmeans.fit(embeddings)
        pred_labels = kmeans.labels_
        score = adjusted_mutual_info_score(data["labels"], pred_labels)
        return score

    @torch.no_grad()
    def linear_svm_classification(self, train_data, test_data, max_iter=100000):
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=max_iter, tol=1e-3))
        print("Training Linear SVM...")
        ret = clf.fit(train_data["embeddings"], train_data["labels"])
        pred_labels = clf.predict(test_data["embeddings"])
        return accuracy_score(test_data["labels"], pred_labels)

    @torch.no_grad()
    def get_embeddings_from_dataloader(self, dataloader):
        self.eval()
        embeddings = []
        outs = []
        labels = []
        filenames = []
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            bg = batch["graph"].to(self.device)
            bg = self._permute_graph_data_channels(bg)
            proj, emb = self.model(bg)
            outs.append(proj.detach().cpu().numpy())
            embeddings.append(emb.detach().cpu().numpy())
            if "label" in batch:
                label = batch["label"]
                labels.append(label.squeeze(-1).detach().cpu().numpy())
            filenames.extend(batch["filename"])
        outs = np.concatenate(outs)
        embeddings = np.concatenate(embeddings)
        if len(labels) > 0:
            labels = np.concatenate(labels)
        else:
            labels = None
        data_count = len(dataloader.dataset)
        assert len(embeddings) == data_count, f"{embeddings.shape}, {data_count}"
        assert len(embeddings.shape) == 2, f"{embeddings.shape}"
        assert len(outs) == data_count, f"{outs.shape}, {data_count}"
        assert len(outs.shape) == 2, f"{outs.shape}"
        if labels is not None:
            assert len(labels) == data_count
            assert len(labels.shape) == 1, f"{labels.shape}"
            assert len(labels.shape) == 1, f"{labels.shape}"
        assert len(filenames) == data_count, f"{len(filenames)}, {data_count}"
        return {"embeddings": embeddings, "labels": labels, "outputs": outs, "filenames": filenames}


class JointPrediction(pl.LightningModule):
    def __init__(
            self,
            input_features=["entity_types", "area", "length", "points", "normals", "tangents", "trimming_mask"],
            emb_dim=384,
            n_head=8,
            n_layer_gat=2,
            n_layer_sat=2,
            n_layer_cat=2,
            bias=False,
            dropout=0.0,
            lr=1e-3
    ):
        super().__init__()
        # if pretrained_model is None:
        #     self.pre_trained_model = Contrastive(
        #         latent_dim=latent_dim,
        #         crv_emb_dim=node_emb_dim,
        #         srf_emb_dim=node_emb_dim,
        #         graph_emb_dim=2*node_emb_dim,
        #         out_dim=out_dim,
        #         channels=channels
        #     ).model
        # else:
        #     self.pre_trained_model = Contrastive.load_from_checkpoint(
        #         latent_dim=latent_dim,
        #         crv_emb_dim=node_emb_dim,
        #         srf_emb_dim=node_emb_dim,
        #         graph_emb_dim=2*node_emb_dim,
        #         out_dim=out_dim,
        #         channels=channels,
        #         checkpoint_path=pretrained_model
        #     ).model
        # self.projection_layer = nn.Sequential(
        #     nn.Linear(2*node_emb_dim, latent_dim, bias=False),
        #     nn.BatchNorm1d(latent_dim),
        #     nn.Dropout(0.3),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, latent_dim, bias=False),
        #     nn.BatchNorm1d(latent_dim),
        #     nn.ReLU(),
        #     nn.Linear(latent_dim, 1, bias=False),
        #     nn.Sigmoid()
        # )
        self.model = JoinABLe(
            input_features,
            emb_dim,
            n_head,
            n_layer_gat,
            n_layer_sat,
            n_layer_cat,
            bias,
            dropout
        )
        self.lr = lr
        self.test_step_outputs = []
        self.test_results = None
        self.criterion = nn.BCELoss(reduction='mean')
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def precision_at_top_k(self, logits, labels, n1, n2, k=None):
        logits = logits.view(n1, n2)
        labels = labels.view(n1, n2)
        if k is None:
            k = metrics.get_k_sequence()
        return metrics.hit_at_top_k(logits, labels, k=k)

    def _permute_graph_data_channels(self, graph):
        graph.ndata["x"] = graph.ndata["x"].permute(0, 3, 1, 2)
        if graph.edata["x"].numel() != 0:
            graph.edata["x"] = graph.edata["x"].permute(0, 2, 1)
        return graph

    def step(self, batch):
        g1, g2, joint_judge, _ = batch
        device = next(self.model.parameters()).device
        g1, g2 = g1.to(device), g2.to(device)
        # g1 = self._permute_graph_data_channels(g1)
        # g2 = self._permute_graph_data_channels(g2)
        # _, global_emb1, _ = self.pre_trained_model(g1)
        # _, global_emb2, _ = self.pre_trained_model(g2)
        # x = self.projection_layer(global_emb1 + global_emb2).squeeze(-1)
        nodes_num = 0
        edge_index_1 = 0
        edge_index_2 = 0
        bg1_node = {
            "uv": g1.ndata["uv"],
            "type": g1.ndata["type"],
            "area": g1.ndata["area"],
        }
        bg1_edge = {
            "uv": g1.edata["uv"],
            "type": g1.edata["type"],
            "length": g1.edata["length"],
        }
        bg2_node = {
            "uv": g2.ndata["uv"],
            "type": g2.ndata["type"],
            "area": g2.ndata["area"],
        }
        bg2_edge = {
            "uv": g2.edata["uv"],
            "type": g2.edata["type"],
            "length": g2.edata["length"],
        }
        x = self.model(nodes_num, edge_index_1, edge_index_2, bg1_node, bg1_edge, bg2_node, bg2_edge)
        loss = self.criterion(x, joint_judge.float())
        return loss, x

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        loss, x = self.step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        train_acc = self.train_accuracy(x, batch[2])
        self.log("train_acc", train_acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, x = self.step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        val_acc = self.val_accuracy(x, batch[2])
        self.log("val_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Get the split we are using from the dataset
        split = self.test_dataloader()[0].dataset.split

        # Inference
        loss, x = self.step(batch)

        # Calculate the precision at k with a default sequence of k
        _, _, joint_judge, file_name = batch
        test_acc = self.test_accuracy(x, joint_judge)
        self.log(f"eval_{split}_acc", test_acc, on_step=False, on_epoch=True, sync_dist=True)

        self.test_step_outputs.append({
            "file_names": file_name,
            "loss": loss.item(),
            "pred": x.item(),
            "label": joint_judge.item()
        })
        return loss

    def on_test_epoch_end(self):
        # Log results to a csv file
        result = []
        for x in self.test_step_outputs:
            result.append((x["file_names"], f"{x['loss']:.2f}", f"{x['pred']:.2f}", x["label"]))
        result.sort(key=lambda x: -float(x[1]))
        self.test_results = result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
