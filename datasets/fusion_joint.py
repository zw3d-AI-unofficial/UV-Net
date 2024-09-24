from datasets.base import BaseDataset
from datasets import util
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import pathlib
import torch
import json
import time
import dgl

class JointBaseDataset(BaseDataset):
    @staticmethod
    def num_classes():
        pass
    
    def __init__(
        self, 
        root_dir, 
        split="train",
        random_rotate=False,
        seed=42
    ):
        if isinstance(root_dir, pathlib.Path):
            self.root_dir = root_dir
        else:
            self.root_dir = pathlib.Path(root_dir)
        assert split in ("train", "val", "validation", "test", "mix_test")
        self.split = split
        self.random_rotate = random_rotate
        self.seed = seed
        # The number of files in the original dataset split
        self.original_file_count = 0
        # The joint set files
        self.files = []

    def __len__(self):
        return len(self.files)

    def get_joint_files(self):
        """Get the joint files to load"""
        all_joint_files = self.get_all_joint_files()
        # Create the train test split
        joint_files = self.get_split(all_joint_files)
        # Store the original file count
        # to keep track of the number of files we filter
        # from the official train/test split
        self.original_file_count = len(joint_files)
        print(f"Loading {len(joint_files)} {self.split} data")
        return joint_files

    def get_all_joint_files(self):
        """Get all the json joint files that look like joint_set_00025.json"""
        pattern = "joint_set_[0-9][0-9][0-9][0-9][0-9].json"
        joint_folder = self.root_dir / "joint"
        return [f.name for f in joint_folder.glob(pattern)]

    def get_split(self, all_joint_files):
        """Get the train/test split"""
        # First check if we have the official split in the dataset dir
        split_file = self.root_dir / "train_test.json"
        if not split_file.exists():
            joint_folder = self.root_dir / "joint"
            split_file = joint_folder / "train_test.json"
        if split_file.exists():
            print("Using official train test split")
            train_joints = []
            val_joints = []
            test_joints = []
            with open(split_file, encoding="utf8") as f:
                official_split = json.load(f)
            if self.split == "train":
                joint_files = official_split["train"]
            elif self.split == "val" or self.split == "validation":
                joint_files = official_split["validation"]
            elif self.split == "test":
                joint_files = official_split["test"]
            elif self.split == "mix_test":
                if "mix_test" not in official_split:
                    raise Exception("Mix test split missing")
                else:
                    joint_files = official_split["mix_test"]
            else:
                raise Exception("Unknown split name")
            joint_files = [f"{f}.json" for f in joint_files]
            return joint_files
        else:
            # We don't have an official split, so we make one
            print("Using new train test split")
            if self.split != "all":
                trainval_joints, test_joints = train_test_split(
                    all_joint_files, test_size=0.2, random_state=self.seed,
                )
                train_joints, val_joints = train_test_split(
                    trainval_joints, test_size=0.25, random_state=self.seed + self.seed,
                )
            if self.split == "train":
                joint_files = train_joints
            elif self.split == "val" or self.split == "validation":
                joint_files = val_joints
            elif self.split == "test":
                joint_files = test_joints
            elif self.split == "all":
                joint_files = all_joint_files
            else:
                raise Exception("Unknown split name")
            return joint_files


class JointGraphDataset(JointBaseDataset):
    SURFACE_GEOM_FEAT_MAP = {
        "type": 6,
        "parameter": 2,
        "axis": 6,
        "box": 6,
        "area": 1,
        "circumference": 1
    }

    CURVE_GEOM_FEAT_MAP = {
        "type": 4,
        "parameter": 2,
        "axis": 6,
        "box": 6,
        "length": 1
    }

    def __init__(
        self,
        root_dir,
        split="train",
        random_rotate=False,
        seed=42,
        center_and_scale=True,
        max_node_count=0,
        channels="points,normals,tangents,trimming_mask"
    ):
        """
        Load the Fusion 360 Gallery joints dataset from graph data
        :param root_dir: Root path to the dataset
        :param split: string Either train, val, test, mix_test, or all set
        :param random_rotate: bool Randomly rotate the point features
        :param seed: Random seed to use
        :param center_and_scale: bool Center and scale the point features
        :param max_node_count: int Exclude joints with more than this number of combined graph nodes
        :param label_scheme: Label remapping scheme.
                Must be one of None, off, ambiguous_on, hole_on, ambiguous_hole_on
        :param channels: Input features to use as a string separated by commas. Can include:
                points, normals, trimming_mask, entity_types, is_face, area, length,
                face_reversed, edge_reversed, reversed, convexity, dihedral_angle"
        """
        super().__init__(
            root_dir,
            split=split,
            random_rotate=random_rotate,
            seed=seed
        )
        self.max_node_count = max_node_count

        with open(self.root_dir / "joint" / "joint_judge.json", 'r') as f:
            self.joint_judge_dict = json.load(f)

        # The graphs as a (g1, g2, joint_graph) triple
        self.graphs = []
        # The graph file used to load g1 and g2
        self.graph_files = []
        # Parse the input features requested
        self.channels = channels.split(',')

        # Get the joint files for our split
        joint_files = self.get_joint_files()

        start_time = time.time()
        for joint_file_name in tqdm(joint_files):
            gs = self.load_joint(joint_file_name, center_and_scale)
            if gs is None:
                continue
            graph1, graph2, joint_judge, joint_file, body_one, body_two = gs
            self.files.append(joint_file.name)
            self.graphs.append([graph1, graph2, joint_judge])
            self.graph_files.append([
                body_one,
                body_two
            ])
        self.convert_to_float32()

        print(f"Total graph load time: {time.time() - start_time} sec")
        skipped_file_count = len(joint_files) - len(self.files)
        print(f"Skipped: {skipped_file_count} files")
        print(f"Done loading {len(self.graphs)} files")

    def convert_to_float32(self):
        for i in range(len(self.graphs)):
            self.graphs[i][0].ndata["uv"] = self.graphs[i][0].ndata["uv"].type(torch.FloatTensor)
            self.graphs[i][0].edata["uv"] = self.graphs[i][0].edata["uv"].type(torch.FloatTensor)
            self.graphs[i][1].ndata["uv"] = self.graphs[i][1].ndata["uv"].type(torch.FloatTensor)
            self.graphs[i][1].edata["uv"] = self.graphs[i][1].edata["uv"].type(torch.FloatTensor)

    def __getitem__(self, idx):
        graph1_name, graph2_name = self.graph_files[idx]
        joint_graph_name = self.files[idx]
        graph1, graph2, joint_judge = self.graphs[idx]
        if self.random_rotate:
            rotation = util.get_random_rotation()
            graph1.ndata["uv"] = util.rotate_uvgrid(graph1.ndata["uv"], rotation)
            graph1.edata["uv"] = util.rotate_uvgrid(graph1.edata["uv"], rotation)
            graph1.ndata["axis"] = util.rotate_uvgrid(graph1.ndata["axis"], rotation)
            graph1.edata["axis"] = util.rotate_uvgrid(graph1.edata["axis"], rotation)
            graph1.ndata["box"] = util.rotate_uvgrid(graph1.ndata["box"], rotation)
            graph1.edata["box"] = util.rotate_uvgrid(graph1.edata["box"], rotation)

            rotation = util.get_random_rotation()
            graph2.ndata["uv"] = util.rotate_uvgrid(graph2.ndata["uv"], rotation)
            graph2.edata["uv"] = util.rotate_uvgrid(graph2.edata["uv"], rotation)
            graph2.ndata["axis"] = util.rotate_uvgrid(graph2.ndata["axis"], rotation)
            graph2.edata["axis"] = util.rotate_uvgrid(graph2.edata["axis"], rotation)
            graph2.ndata["box"] = util.rotate_uvgrid(graph2.ndata["box"], rotation)
            graph2.edata["box"] = util.rotate_uvgrid(graph2.edata["box"], rotation)
        return [graph1, graph2, joint_judge, (graph1_name, graph2_name, joint_graph_name)]

    def _collate(self, batch):
        batched_graph1 = dgl.batch([x[0] for x in batch])
        batched_graph2 = dgl.batch([x[1] for x in batch])
        if batched_graph1.num_nodes() + batched_graph2.num_nodes() > 16384:
            print("skip")
            return None
        batched_joint_judge = torch.tensor([x[2] for x in batch], dtype=torch.int32)
        batch_file_names = [x[3] for x in batch]
        return batched_graph1, batched_graph2, batched_joint_judge, batch_file_names

    def get_dataloader(self, batch_size=1, shuffle=True, num_workers=0, drop_last=True):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=True
        )
    
    @staticmethod
    def get_bounding_box(inp):
        pts = inp[:, :, :, :3].reshape((-1, 3))
        mask = inp[:, :, :, 6].reshape(-1)
        point_indices_inside_faces = mask == 1
        pts = pts[point_indices_inside_faces, :]
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
        return torch.tensor(box)

    @staticmethod
    def get_center_and_scale(grid1, grid2):
        bbox1 = JointGraphDataset.get_bounding_box(grid1)
        bbox2 = JointGraphDataset.get_bounding_box(grid2)
        # bbox_min = torch.minimum(bbox1[0], bbox2[0])
        # bbox_max = torch.maximum(bbox1[1], bbox2[1])
        # span = bbox_max - bbox_min
        # max_span = torch.max(span)
        # scale = 2.0 / max_span
        bboxes = torch.cat((bbox1, bbox2))
        center1 = 0.5 * (bbox1[0] + bbox1[1])
        center2 = 0.5 * (bbox2[0] + bbox2[1])
        scale = (1.0 / bboxes.abs().max()) * 0.999999
        return center1, center2, scale

    def scale_features(self, g1, g2):
        """Scale the points for both graphs"""
        # Get the combined bounding box
        center1, center2, scale = JointGraphDataset.get_center_and_scale(g1.ndata["uv"], g2.ndata["uv"])
        g1.ndata["uv"][:, :, :, :3] -= center1
        g1.ndata["uv"][:, :, :, :3] *= scale
        # Check we aren't too far out of bounds due to the masked surface
        if torch.max(g1.ndata["uv"][:, :, :, :3]) > 2.0 or torch.max(g1.ndata["uv"][:, :, :, :3]) < -2.0:
            return False
        g2.ndata["uv"][:, :, :, :3] -= center2
        g2.ndata["uv"][:, :, :, :3] *= scale
        if torch.max(g2.ndata["uv"][:, :, :, :3]) > 2.0 or torch.max(g2.ndata["uv"][:, :, :, :3]) < -2.0:
            return False
        
        g1.ndata["parameter"][:, 0] *= scale
        g1.edata["parameter"][:, 0] *= scale
        g1.ndata["axis"][:, :3] -= center1
        g1.ndata["axis"][:, :3] *= scale
        g1.edata["axis"][:, :3] -= center1
        g1.edata["axis"][:, :3] *= scale        
        g1.ndata["box"][:, :3] -= center1
        g1.ndata["box"][:, :3] *= scale
        g1.edata["box"][:, :3] -= center1
        g1.edata["box"][:, :3] *= scale        
        g1.ndata["box"][:, 3:] -= center1
        g1.ndata["box"][:, 3:] *= scale
        g1.edata["box"][:, 3:] -= center1
        g1.edata["box"][:, 3:] *= scale
        g1.ndata["area"] *= scale * scale
        g1.ndata["circumference"] *= scale
        g1.edata["length"] *= scale
        
        g2.ndata["parameter"][:, 0] *= scale
        g2.edata["parameter"][:, 0] *= scale
        g2.ndata["axis"][:, :3] -= center2
        g2.ndata["axis"][:, :3] *= scale
        g2.edata["axis"][:, :3] -= center2
        g2.edata["axis"][:, :3] *= scale        
        g2.ndata["box"][:, :3] -= center2
        g2.ndata["box"][:, :3] *= scale
        g2.edata["box"][:, :3] -= center2
        g2.edata["box"][:, :3] *= scale        
        g2.ndata["box"][:, 3:] -= center2
        g2.ndata["box"][:, 3:] *= scale
        g2.edata["box"][:, 3:] -= center2
        g2.edata["box"][:, 3:] *= scale
        g2.ndata["area"] *= scale * scale
        g2.ndata["circumference"] *= scale
        g2.edata["length"] *= scale

        return True
    
    def load_joint_judge_lb(self, joint_judge_dict, joint_file_name):
        joint_file_name = joint_file_name[:joint_file_name.find('.')]
        # 从 joint_judge_dict 中获取对应的标签
        label = joint_judge_dict.get(joint_file_name, None)
        if label is None:
            raise ValueError(f"Label not found for joint file name: {joint_file_name}")

        if label == 0 or label == 1:
            return label
        else:
            raise ValueError(f"Invalid label value: {label} for joint file name: {joint_file_name}")

    def load_joint(self, joint_file_name, center_and_scale=True):
        """Load a joint file and return a graph"""
        joint_folder = self.root_dir / "joint"
        joint_file = joint_folder / joint_file_name
        with open(joint_file, encoding="utf8") as f:
            joint_data = json.load(f)
        g1, face_count1, edge_count1 = self.load_part(
            joint_data["body_one"])
        if g1 is None or edge_count1 == 0:
            return None
        g2, face_count2, edge_count2 = self.load_part(
            joint_data["body_two"])
        if g2 is None or edge_count2 == 0:
            return None
        # Limit the maximum number of combined nodes
        total_nodes = face_count1 + face_count2
        if self.max_node_count > 0:
            if total_nodes > self.max_node_count:
                return None
        joint_judge = self.load_joint_judge_lb(self.joint_judge_dict, joint_file.name)
        # Scale geometry features from both graphs with a common scale
        if center_and_scale:
            scale_good = self.scale_features(g1, g2)
            # Throw out if we can't scale properly due to the masked surface area
            if not scale_good:
                print("Discarding graph with bad scale")
                return None
        return g1, g2, joint_judge, joint_file, joint_data["body_one"], joint_data["body_two"]

    def load_part(self, body):
        """Load a graph created from a brep body"""
        part_folder = self.root_dir / "graph"
        sample = self.load_one_graph(part_folder / (body  + ".bin"))
        graph = sample["graph"]
        return graph, graph.number_of_nodes(), graph.number_of_edges()

    def offset_joint_index(self, entity_index, entity_type, face_count, entity_count):
        """Offset the joint index for the label matrix"""
        joint_index = entity_index
        if entity_type == "BRepEdge":
            # If this is a brep edge we need to increment the index
            # to start past the number of faces as those are stored first
            joint_index += face_count
        # If we have a BRepFace life is simple...
        assert joint_index >= 0
        return joint_index

