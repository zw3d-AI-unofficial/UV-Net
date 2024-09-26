import torch
from pathlib import Path
from args import args_train
# from train import JointPrediction
from datasets.joint_graph_dataset import JointGraphDataset
from uvnet.joinable import JoinABLe
import onnx

def export_to_onnx(args, model):
    num_nodes1, num_edges1, num_nodes2, num_edges2 = 5, 5, 10, 10
    dummy_g1_node = torch.randn(num_nodes1, JointGraphDataset.grid_size, JointGraphDataset.grid_size, JointGraphDataset.grid_channels)
    dummy_g1_edge = torch.randn(num_edges1, JointGraphDataset.grid_size, JointGraphDataset.grid_channels - 1)
    # dummy_g1_ent = torch.randn(num_nodes1, JointGraphDataset.ent_feature_size)
    # dummy_g1_ent[:, 1] = torch.randint(0, len(JointGraphDataset.curve_type_map), [num_nodes1])
    dummy_g1_edge_index = torch.randint(0, num_nodes1, (2, num_edges1))
    dummy_g2_node = torch.randn(num_nodes2, JointGraphDataset.grid_size, JointGraphDataset.grid_size, JointGraphDataset.grid_channels)
    dummy_g2_edge = torch.randn(num_edges2, JointGraphDataset.grid_size, JointGraphDataset.grid_channels - 1)
    # dummy_g2_ent = torch.randn(num_nodes2, JointGraphDataset.ent_feature_size)
    # dummy_g2_ent[:, 1] = torch.randint(0, len(JointGraphDataset.curve_type_map), [num_nodes2])
    dummy_g2_edge_index = torch.randint(0, num_nodes2, (2, num_edges2))

    dummy_num_nodes = torch.tensor([[num_nodes1], [num_nodes2]])
    dummy_g1_node_type = torch.randint(0, 6, [num_nodes1])
    dummy_g1_node_area = torch.randn(num_nodes1, 1)
    dummy_g1_edge_type = torch.randint(0, 4, [num_edges1])
    dummy_g1_edge_length = torch.randn(num_edges1, 1)
    dummy_g2_node_type = torch.randint(0, 6, [num_nodes2])
    dummy_g2_node_area = torch.randn(num_nodes2, 1)
    dummy_g2_edge_type = torch.randint(0, 4, [num_edges2])
    dummy_g2_edge_length = torch.randn(num_edges2, 1)
    input = (
        dummy_num_nodes,
        dummy_g1_edge_index,
        dummy_g2_edge_index,
        # node_data_1
        dummy_g1_node,
        dummy_g1_node_type,
        dummy_g1_node_area,
        # edge_data_1
        dummy_g1_edge,
        dummy_g1_edge_type,
        dummy_g1_edge_length,
        # node_data_2
        dummy_g2_node,
        dummy_g2_node_type,
        dummy_g2_node_area,
        # edge_data_2
        dummy_g2_edge,
        dummy_g2_edge_type,
        dummy_g2_edge_length,
    )

    input_names = [
        "dummy_num_nodes",
        "dummy_g1_edge_index",
        "dummy_g2_edge_index",
        # node_data_1
        "dummy_g1_node",
        "dummy_g1_node_type",
        "dummy_g1_node_area",
        # edge_data_1
        "dummy_g1_edge",
        "dummy_g1_edge_type",
        "dummy_g1_edge_length",
        # node_data_2
        "dummy_g2_node",
        "dummy_g2_node_type",
        "dummy_g2_node_area",
        # edge_data_2
        "dummy_g2_edge",
        "dummy_g2_edge_type",
        "dummy_g2_edge_length",
    ]

    dynamic_axes = {
        'dummy_g1_edge_index': {1: 'num_edges1'},
        'dummy_g2_edge_index': {1: 'num_edges2'},
        'dummy_g1_node': {0: 'num_nodes1'},
        'dummy_g2_node': {0: 'num_nodes2'},
        'dummy_g1_edge': {0: 'num_edges1'},
        'dummy_g2_edge': {0: 'num_edges2'},
        "dummy_g1_node_type": {0: 'num_nodes1'},
        "dummy_g1_node_area": {0: 'num_nodes1'},
        "dummy_g1_edge_type": {0: 'num_edges1'},
        "dummy_g1_edge_length": {0: 'num_edges1'},
        "dummy_g2_node_type": {0: 'num_nodes2'},
        "dummy_g2_node_area": {0: 'num_nodes2'},
        "dummy_g2_edge_type": {0: 'num_edges2'},
        "dummy_g2_edge_length": {0: 'num_edges2'},
    }

    # exp_dir = Path(args.exp_dir)
    # exp_name_dir = exp_dir / args.exp_name
    file = "/home/xuhaizi/UV-Net/test_19_no_bias_small.onnx"
    model.eval()
    # model.double()
    torch.onnx.export(
        model,
        input,
        file,
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=input_names,
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    onnx_model = onnx.load(file)
    onnx.checker.check_model(onnx_model, full_check=True)
    print("The model is successfully exported and is valid.")

if __name__ == "__main__":
    args = args_train.get_args()
    
    exp_dir = Path(args.exp_dir)
    exp_name_dir = exp_dir / args.exp_name
    # checkpoint_file = exp_name_dir / f"{args.checkpoint}.ckpt"
    # model = JointPrediction.load_from_checkpoint(checkpoint_file).model
    model = JoinABLe(
        input_features=["type","area","length","points","normals","tangents","trimming_mask"],
        emb_dim=64,
        n_head=8,
        n_layer_gat=1,
        n_layer_sat=0,
        n_layer_cat=0,
        bias=False,
        dropout=0.0
    )
    model.eval()
    model.cpu()
    export_to_onnx(args, model)