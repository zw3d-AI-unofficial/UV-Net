import torch
from pathlib import Path
from args import args_train
from uvnet.models import JointPrediction
from uvnet.joinable import JoinABLe
import onnx

def export_to_onnx(model):
    num_nodes1, num_edges1, num_nodes2, num_edges2, grid_size, grid_channel = 5, 6, 10, 12, 10, 7
    num_nodes = torch.tensor([[num_nodes1], [num_nodes2]])
    g1_edge_index = torch.randint(0, num_nodes1, (2, num_edges1))
    g2_edge_index = torch.randint(0, num_nodes2, (2, num_edges2))
    g1_node_uv = torch.randn(num_nodes1, grid_size, grid_size, grid_channel)
    g1_node_type = torch.randint(0, 6, [num_nodes1])
    g1_node_area = torch.randn(num_nodes1, 1)
    g1_edge_uv = torch.randn(num_edges1, grid_size, grid_channel - 1)
    g1_edge_type = torch.randint(0, 4, [num_edges1])
    g1_edge_length = torch.randn(num_edges1, 1)
    g2_node_uv = torch.randn(num_nodes2, grid_size, grid_size, grid_channel)
    g2_node_type = torch.randint(0, 6, [num_nodes2])
    g2_node_area = torch.randn(num_nodes2, 1)
    g2_edge_uv = torch.randn(num_edges2, grid_size, grid_channel - 1)
    g2_edge_type = torch.randint(0, 4, [num_edges2])
    g2_edge_length = torch.randn(num_edges2, 1)
    jg_edge_index = torch.randint(0, max(num_nodes1, num_nodes2), (2, num_nodes1 * num_nodes2))
    input = (
        num_nodes,
        g1_edge_index,
        g2_edge_index,
        g1_node_uv,
        g1_node_type,
        g1_node_area,
        g1_edge_uv,
        g1_edge_type,
        g1_edge_length,
        g2_node_uv,
        g2_node_type,
        g2_node_area,
        g2_edge_uv,
        g2_edge_type,
        g2_edge_length,
        jg_edge_index
    )

    input_names = [
        "num_nodes",
        "g1_edge_index",
        "g2_edge_index",
        "g1_node",
        "g1_node_type",
        "g1_node_area",
        "g1_edge",
        "g1_edge_type",
        "g1_edge_length",
        "g2_node",
        "g2_node_type",
        "g2_node_area",
        "g2_edge",
        "g2_edge_type",
        "g2_edge_length",
        'jg_edge_index'
    ]

    dynamic_axes = {
        'g1_edge_index': {1: 'num_edges1'},
        'g2_edge_index': {1: 'num_edges2'},
        'g1_node': {0: 'num_nodes1'},
        'g2_node': {0: 'num_nodes2'},
        'g1_edge': {0: 'num_edges1'},
        'g2_edge': {0: 'num_edges2'},
        "g1_node_type": {0: 'num_nodes1'},
        "g1_node_area": {0: 'num_nodes1'},
        "g1_edge_type": {0: 'num_edges1'},
        "g1_edge_length": {0: 'num_edges1'},
        "g2_node_type": {0: 'num_nodes2'},
        "g2_node_area": {0: 'num_nodes2'},
        "g2_edge_type": {0: 'num_edges2'},
        "g2_edge_length": {0: 'num_edges2'},
        'jg_edge_index': {1: 'num_nodes1 * num_nodes2'},
        'output': {0: 'num_nodes1 * num_nodes2'}
    }

    file = "/home/fusiqiao/Projects/UV-Net/basic.onnx"
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
    checkpoint_file = Path(args.checkpoint)
    model = JointPrediction.load_from_checkpoint(emb_dim=384, checkpoint_path=checkpoint_file).model
    # model = JoinABLe()
    model.eval()
    model.cpu()
    export_to_onnx(model)