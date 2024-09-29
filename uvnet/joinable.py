import torch
import torch.nn as nn
import torch.nn.functional as F
import uvnet.encoders
from datasets.fusion_joint import JointGraphDataset
from dgl.nn import GATv2Conv


def cnn2d(inp_channels, hidden_channels, out_dim, num_layers=1):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Conv2d(inp_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=3, padding=1))
        modules.append(nn.ELU())
    modules.append(nn.AdaptiveAvgPool2d(1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(hidden_channels, out_dim, bias=True))
    return nn.Sequential(*modules)


def cnn1d(inp_channels, hidden_channels, out_dim, num_layers=1):
    assert num_layers >= 1
    modules = []
    for i in range(num_layers - 1):
        modules.append(nn.Conv1d(inp_channels if i == 0 else hidden_channels, hidden_channels, kernel_size=3, padding=1))
        modules.append(nn.ELU())
    modules.append(nn.AdaptiveAvgPool1d(1))
    modules.append(nn.Flatten())
    modules.append(nn.Linear(hidden_channels, out_dim, bias=True))
    return nn.Sequential(*modules)


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, bias=False, dropout=0.0):
        super().__init__()
        self.c_fc    = nn.Linear(input_dim, 4 * input_dim, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * input_dim, output_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class LayerNorm(nn.Module):

    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, n_dim, bias=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_dim))
        self.bias = nn.Parameter(torch.zeros(n_dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class FaceEncoder(nn.Module):

    def __init__(
            self, 
            input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=384,
            bias=False,
            dropout=0.0
        ):
        super().__init__()
        # Calculate the total size of each feature list
        self.grid_features = []
        if "points" in input_features:
            self.grid_features.extend([0, 1, 2])
        if "normals" in input_features:
            self.grid_features.extend([3, 4, 5])
        if "trimming_mask" in input_features:
            self.grid_features.append(6)
        self.geom_features = []
        self.geom_feature_size = 0
        for feat, feat_size in JointGraphDataset.SURFACE_GEOM_FEAT_MAP.items():
            if feat in input_features:
                self.geom_features.append(feat)
                self.geom_feature_size += feat_size

        # Setup the layers
        self.emb_dim = emb_dim
        if len(self.grid_features) > 0:
            self.grid_embd = cnn2d(len(self.grid_features), emb_dim, emb_dim, num_layers=3)
        if self.geom_feature_size > 0:
            self.geom_embd = nn.Linear(self.geom_feature_size, emb_dim)
        self.ln = LayerNorm(emb_dim, bias=bias)
        self.mlp = MLP(emb_dim, emb_dim, bias, dropout)

    def forward(self, bg):
        x = torch.zeros(bg.num_nodes(), self.emb_dim, dtype=torch.float, device=bg.ndata['uv'].device)
        if len(self.grid_features) > 0:  
            grid_feat = bg.ndata['uv'][:, :, :, self.grid_features].permute(0, 3, 1, 2)
            x += self.grid_embd(grid_feat)
        if self.geom_feature_size > 0:
            geom_feat = []
            for feat in self.geom_features:
                if feat == "type":
                    feat = F.one_hot(bg.ndata[feat], num_classes=JointGraphDataset.SURFACE_GEOM_FEAT_MAP["type"])
                else:
                    feat = bg.ndata[feat]
                    if len(feat.shape) == 1:
                        feat = feat.unsqueeze(1)
                geom_feat.append(feat)
            x += self.geom_embd(torch.cat(geom_feat, dim=1).float())
        x = x + self.mlp(self.ln(x))
        return x


class EdgeEncoder(nn.Module):

    def __init__(
            self, 
            input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=384,
            bias=False,
            dropout=0.0
        ):
        super().__init__()
        # Calculate the total size of each feature list
        self.grid_features = []
        if "points" in input_features:
            self.grid_features.extend([0, 1, 2])
        if "tangents" in input_features:
            self.grid_features.extend([3, 4, 5])
        self.geom_features = []
        self.geom_feature_size = 0
        for feat, feat_size in JointGraphDataset.CURVE_GEOM_FEAT_MAP.items():
            if feat in input_features:
                self.geom_features.append(feat)
                self.geom_feature_size += feat_size

        # Setup the layers
        self.emb_dim = emb_dim
        if len(self.grid_features) > 0:
            self.grid_embd = cnn1d(len(self.grid_features), emb_dim, emb_dim, num_layers=3)
        if self.geom_feature_size > 0:
            self.geom_embd = nn.Linear(self.geom_feature_size, emb_dim)
        self.ln = LayerNorm(emb_dim, bias)
        self.mlp = MLP(emb_dim, emb_dim, bias, dropout)
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            torch.nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, bg):
        x = torch.zeros(bg.num_edges(), self.emb_dim, dtype=torch.float, device=bg.edata['uv'].device)
        if len(self.grid_features) > 0:  
            grid_feat = bg.edata['uv'][:, :, self.grid_features].permute(0, 2, 1)
            x += self.grid_embd(grid_feat) 
        if self.geom_feature_size > 0:
            geom_feat = []
            for feat in self.geom_features:
                if feat == "type":
                    feat = F.one_hot(bg.edata[feat], num_classes=JointGraphDataset.CURVE_GEOM_FEAT_MAP["type"])
                else:
                    feat = bg.edata[feat]
                    if len(feat.shape) == 1:
                        feat = feat.unsqueeze(1)
                geom_feat.append(feat)
            x += self.geom_embd(torch.cat(geom_feat, dim=1).float())
        x = x + self.mlp(self.ln(x))
        return x


class SelfAttention(nn.Module):

    def __init__(self, emb_dim, n_head=8, bias=False, dropout=0.0):
        super().__init__()
        assert emb_dim % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = emb_dim
        self.dropout = dropout

    def forward(self, x, attn_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) nh*hs = 384 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) torch.Size([1, 8, 36, 48])
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # self-attention: 
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class SATBlock(nn.Module):

    def __init__(self, emb_dim, n_head=8, bias=False, dropout=0.0):
        super().__init__()
        self.ln_1 = LayerNorm(emb_dim, bias=bias)
        self.attn = SelfAttention(emb_dim, n_head, bias, dropout)
        self.ln_2 = LayerNorm(emb_dim, bias=bias)
        self.mlp = MLP(emb_dim, emb_dim, bias, dropout)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    

class CustomGATv2Conv(GATv2Conv):

    def __init__(self, in_feats, out_feats, n_head, dropout=0.0):
        super(CustomGATv2Conv, self).__init__(in_feats, out_feats, n_head, feat_drop=dropout, attn_drop=dropout, allow_zero_in_degree=True)

    def edge_attention(self, edges):
        edge_weights = edges.data['edge_attr']
        return {'e': self.leaky_relu((edges.src['z'] * edges.dst['z'] + edge_weights).sum(dim=-1))}

    def forward(self, graph, node_features, edge_attr):
        graph.edata['edge_attr'] = edge_attr
        return super(CustomGATv2Conv, self).forward(graph, node_features)


class GATBlock(nn.Module):
    
    def __init__(self, emb_dim, n_head=8, bias=False, dropout=0.0):
        super().__init__()
        self.ln_1 = LayerNorm(emb_dim, bias)
        self.attn = CustomGATv2Conv(emb_dim, emb_dim // n_head, n_head, dropout)
        self.ln_2 = LayerNorm(emb_dim, bias)
        self.mlp = MLP(emb_dim, emb_dim, bias, dropout)
        self.emb_dim = emb_dim

    def forward(self, graph, x, edge_attr):
        x = x + self.attn(graph, self.ln_1(x), edge_attr).reshape(-1, self.emb_dim)
        x = x + self.mlp(self.ln_2(x))
        return x      
    

class GraphEncoder(nn.Module):

    def __init__(
            self, 
            input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=384,
            n_head=8,
            n_layer_gat=2,
            n_layer_sat=2,
            bias=False,
            dropout=0.0
        ):
        super().__init__()
        self.face_encoder = FaceEncoder(input_features, emb_dim, bias, dropout)
        self.edge_encoder = EdgeEncoder(input_features, emb_dim, bias, dropout)
        # self.graph_encoder = uvnet.encoders.UVNetGraphEncoder(emb_dim, emb_dim, emb_dim, emb_dim)
        self.gat_list = nn.ModuleList([GATBlock(emb_dim, n_head, bias, dropout) for _ in range(n_layer_gat)])
        self.sat_list = nn.ModuleList([SATBlock(emb_dim, n_head, bias, dropout) for _ in range(n_layer_sat)])
        self.drop = nn.Dropout(dropout)

    def forward(self, bg):
        face_emb = self.drop(self.face_encoder(bg))
        edge_emb = self.drop(self.edge_encoder(bg))    
        for block in self.gat_list:
            face_emb = block(bg, face_emb, edge_emb)
        # Prepare data for attention layers 
        node_counts = bg.batch_num_nodes().tolist()
        x = self.split_and_pad(face_emb, node_counts)                                                         # torch.Size([2, 140, 384])
        # Attention layers
        attn_mask = self.get_attn_mask(x, node_counts)
        for block in self.sat_list:
            x = block(x, attn_mask)
        return x

    def split_and_pad(self, x, node_counts):
        x_list = torch.split(x, node_counts)
        max_node_count = max(node_counts)

        padded_x_list = []
        for x in x_list:
            node_count = x.size(0)
            padding = (0, 0, 0, max_node_count - node_count)
            padded_x = F.pad(x, padding, value=0)
            padded_x_list.append(padded_x)
        return torch.stack(padded_x_list)

    def get_attn_mask(self, x, node_counts):
        B, T, _ = x.shape
        attn_mask = torch.ones((B, 1, T, T), dtype=torch.float32, device=x.device)
        for i in range(B):
            attn_mask[i, 0, :node_counts[i], :node_counts[i]] = 0
        A_SMALL_NUMBER = -1e9
        attn_mask *= A_SMALL_NUMBER
        return attn_mask


class CrossAttention(nn.Module):

    def __init__(self, emb_dim, n_head=8, bias=False, dropout=0.0):
        super().__init__()
        assert emb_dim % n_head == 0                           #确保了嵌入维度 n_embd 能够被头的数量 n_head 整除
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim, bias=bias) #确保了嵌入维度 n_embd 能够被头的数量 n_head 整除
        self.c_attn = nn.Linear(emb_dim, 3 * emb_dim, bias=bias)
        # output projection
        self.c_proj = nn.Linear(emb_dim, emb_dim, bias=bias)     #将注意力的结果映射回原始的嵌入维度
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = emb_dim
        self.dropout = dropout

    def forward(self, x1, x2, attn_mask):
        B, T1, C = x1.size() # batch size, sequence length, embedding dimensionality (n_embd)
        B, T2, C = x2.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q1, k1, v1  = self.c_attn(x1).split(self.n_embd, dim=2)
        k1 = k1.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T1, hs)
        q1 = q1.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T1, hs)
        v1 = v1.view(B, T1, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T1, hs)
        q2, k2, v2  = self.c_attn(x2).split(self.n_embd, dim=2)
        k2 = k2.view(B, T2, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T2, hs)
        q2 = q2.view(B, T2, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T2, hs)
        v2 = v2.view(B, T2, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T2, hs)

        # cross-attention: 
        y1 = torch.nn.functional.scaled_dot_product_attention(q1, k2, v2, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        y1 = y1.transpose(1, 2).contiguous().view(B, T1, C) # re-assemble all head outputs side by side 组装成原始形状
        y1 = self.resid_dropout(self.c_proj(y1))

        attn_mask=attn_mask.transpose(2, 3)
        y2 = torch.nn.functional.scaled_dot_product_attention(q2, k1, v1, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0)
        y2 = y2.transpose(1, 2).contiguous().view(B, T2, C) # re-assemble all head outputs side by side
        y2 = self.resid_dropout(self.c_proj(y2))
        return y1, y2


class CATBlock(nn.Module):

    def __init__(self, emb_dim, n_head=8, bias=False, dropout=0.0):
        super().__init__()
        self.ln_1 = LayerNorm(emb_dim, bias=bias)
        self.attn = CrossAttention(emb_dim, emb_dim // n_head, n_head, dropout)
        self.ln_2 = LayerNorm(emb_dim, bias=bias)
        self.mlp = MLP(emb_dim, emb_dim, bias, dropout)

    def forward(self, x1, x2, attn_mask_cross):
        x = self.attn(self.ln_1(x1), self.ln_1(x2), attn_mask_cross)
        x1 = x1 + x[0]
        x2 = x2 + x[1]
        x1 = x1 + self.mlp(self.ln_2(x1))
        x2 = x2 + self.mlp(self.ln_2(x2))
        return x1, x2


class JointPairHead(nn.Module):

    def __init__(self, input_dim, n_layer, bias=False, dropout=0.0):
        super().__init__()
        self.ln = LayerNorm(input_dim, bias=bias)
        self.projection_layer = nn.ModuleList()
        for _ in range(n_layer):
            self.projection_layer.append(nn.Linear(input_dim, input_dim, bias=bias))
            self.projection_layer.append(nn.BatchNorm1d(input_dim))
            self.projection_layer.append(nn.Dropout(dropout))
            self.projection_layer.append(nn.ReLU())
        self.output_layer = nn.Linear(input_dim, 1, bias)

    def forward(self, x, jg):
        src_nodes, dst_nodes = jg.edges()
        x = x[src_nodes, :] + x[dst_nodes, :]
        x = self.ln(x)
        for layer in self.projection_layer:
            x = layer(x)
        logits = self.output_layer(x).squeeze()
        return logits
    

class JointTypeHead(nn.Module):

    def __init__(self, input_dim, n_layer, bias=False, dropout=0.0):
        super().__init__()
        self.ln = LayerNorm(input_dim, bias=bias)
        self.projection_layer = nn.ModuleList()
        for _ in range(n_layer):
            self.projection_layer.append(nn.Linear(input_dim, input_dim, bias=bias))
            self.projection_layer.append(nn.BatchNorm1d(input_dim))
            self.projection_layer.append(nn.Dropout(dropout))
            self.projection_layer.append(nn.ReLU())
        self.output_layer = nn.Linear(input_dim, len(JointGraphDataset.JOINT_TYPE_MAP), bias)

    def forward(self, x, jg):
        ids = torch.where(jg.edata["label_matrix"] == 1)[0].long()
        src_nodes, dst_nodes = jg.edges()
        x = x[src_nodes[ids], :] + x[dst_nodes[ids], :]
        x = self.ln(x)
        for layer in self.projection_layer:
            x = layer(x)
        logits = self.output_layer(x).squeeze()
        return logits


class JoinABLe(nn.Module):
    def __init__(
            self, 
            input_features=["type","area","length","points","normals","tangents","trimming_mask"],
            emb_dim=384,
            n_head=8,
            n_layer_gat=2,
            n_layer_sat=2,
            n_layer_cat=2,
            n_layer_pair_head=2,
            with_type=False,
            n_layer_type_head=2,
            bias=False,
            dropout=0.0
        ):
        super().__init__()
        self.graph_encoder = GraphEncoder(input_features, emb_dim, n_head, n_layer_gat, n_layer_sat, bias, dropout)
        self.cat_list = nn.ModuleList([CATBlock(emb_dim, n_head, bias, dropout) for _ in range(n_layer_cat)])
        self.drop = nn.Dropout(dropout)
        self.pair_head = JointPairHead(emb_dim, n_layer_pair_head, bias, dropout)
        self.type_head = JointTypeHead(emb_dim, n_layer_type_head, dropout)
        self.with_type = with_type
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
    
    def forward(self, bg1, bg2, jg):
        x1 = self.graph_encoder(bg1)
        x2 = self.graph_encoder(bg2)
        node_counts1 = bg1.batch_num_nodes()
        node_counts2 = bg2.batch_num_nodes()
        attn_mask = self.get_attn_mask(x1, x2, node_counts1, node_counts2)
        for block in self.cat_list:
            x1, x2 = block(x1, x2, attn_mask)
        # Pass to post-net
        x = self.unpad_and_concat(x1, x2, node_counts1, node_counts2)
        pair_logits = self.pair_head(x, jg)
        type_logits = None
        if self.with_type:
            type_logits = self.type_head(x, jg)
        return pair_logits, type_logits

    def get_attn_mask(self, x1, x2, n_nodes1, n_nodes2):
        B, T1, _ = x1.shape
        B, T2, _ = x2.shape
        attn_mask = torch.ones((B, 1, T1, T2), dtype=torch.float32, device=x1.device)
        for i in range(B):
            attn_mask[i, 0, :n_nodes1[i], :n_nodes2[i]] = 0
        A_SMALL_NUMBER = -1e9
        attn_mask *= A_SMALL_NUMBER
        return attn_mask

    def unpad_and_concat(self, x1, x2, n_nodes1, n_nodes2):
        concat_x = []
        for i in range(len(n_nodes1)):
            size1_i = n_nodes1[i]
            size2_i = n_nodes2[i]
            # Concatenate features from graph1 and graph2 in a interleaved fashion
            # as this is the format that the joint graph expects
            x1_i = x1[i, :size1_i, :]
            x2_i = x2[i, :size2_i, :]
            concat_x.append(x1_i)
            concat_x.append(x2_i)
        x = torch.cat(concat_x, dim=0)
        return x