import onnxruntime as ort
import numpy as np
import torch
from datasets.joint_graph_dataset import JointGraphDataset

# 加载 ONNX 模型
model_path = "/home/xuhaizi/UV-Net/test_19_no_bias_small.onnx"
session = ort.InferenceSession(model_path)

num_nodes1, num_edges1, num_nodes2, num_edges2 = 6, 6, 10, 10
dummy_g1_node = torch.randn(num_nodes1, JointGraphDataset.grid_size, JointGraphDataset.grid_size, JointGraphDataset.grid_channels, dtype=torch.double)
dummy_g1_edge = torch.randn(num_edges1, JointGraphDataset.grid_size, JointGraphDataset.grid_channels - 1, dtype=torch.double)
# dummy_g1_ent = torch.randn(num_nodes1, JointGraphDataset.ent_feature_size)
# dummy_g1_ent[:, 1] = torch.randint(0, len(JointGraphDataset.curve_type_map), [num_nodes1])
dummy_g1_edge_index = torch.randint(0, num_nodes1, (2, num_edges1))
dummy_g2_node = torch.randn(num_nodes2, JointGraphDataset.grid_size, JointGraphDataset.grid_size, JointGraphDataset.grid_channels, dtype=torch.double)
dummy_g2_edge = torch.randn(num_edges2, JointGraphDataset.grid_size, JointGraphDataset.grid_channels - 1, dtype=torch.double)
# dummy_g2_ent = torch.randn(num_nodes2, JointGraphDataset.ent_feature_size)
# dummy_g2_ent[:, 1] = torch.randint(0, len(JointGraphDataset.curve_type_map), [num_nodes2])
dummy_g2_edge_index = torch.randint(0, num_nodes2, (2, num_edges2))

dummy_num_nodes = torch.tensor([[num_nodes1], [num_nodes2]])
dummy_g1_node_type = torch.randint(0, 6, [num_nodes1])
dummy_g1_node_area = torch.randn(num_nodes1, 1, dtype=torch.double)
dummy_g1_edge_type = torch.randint(0, 4, [num_edges1])
dummy_g1_edge_length = torch.randn(num_edges1, 1, dtype=torch.double)
dummy_g2_node_type = torch.randint(0, 6, [num_nodes2])
dummy_g2_node_area = torch.randn(num_nodes2, 1, dtype=torch.double)
dummy_g2_edge_type = torch.randint(0, 4, [num_edges2])
dummy_g2_edge_length = torch.randn(num_edges2, 1, dtype=torch.double)

input_data = {
    "dummy_g1_node": dummy_g1_node.numpy(),
    "dummy_g1_node_type": dummy_g1_node_type.numpy(),
    "dummy_g1_node_area": dummy_g1_node_area.numpy(),

    "dummy_g2_node": dummy_g2_node.numpy(),
    "dummy_g2_node_type": dummy_g2_node_type.numpy(),
    "dummy_g2_node_area": dummy_g2_node_area.numpy(),
}

# 进行推理
output = session.run(['output'], input_data)

# 打印输出
print(torch.topk(torch.tensor(np.array(output)), 5))