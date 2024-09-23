"""

JoinABLe training args

"""

from args import args_common


def get_parser():
    """Return the training args parser"""
    parser = args_common.get_parser()
    parser.add_argument(
        "--input_features",
        type=str,
        default="entity_types,area,length,points,normals,tangents,trimming_mask",
        help="Input features to use as a string separated by commas.\
                Can include: points, normals, tangents, trimming_mask,\
                axis_pos, axis_dir, bounding_box, entity_types\
                area, circumference, param_1, param_2\
                length, radius, start_point, middle_point, end_point"
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Using feature quantization."
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        default=8,
        help="Number of bit of quantization."
    )
    parser.add_argument(
        "--n_layer_gat",
        type=int,
        default=2,
        help="Number of GAT layers."
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
        "--n_layer_head1",
        type=int,
        default=2,
        help="Number of joint prediction layers."
    )
    parser.add_argument(
        "--with_type",
        action="store_true",
        default=False,
        help="Use type head."
    )
    parser.add_argument(
        "--n_layer_head2",
        type=int,
        default=2,
        help="Number of joint type prediction layers."
    )
    parser.add_argument(
        "--n_head",
        type=int,
        default=8,
        help="Number of attention heads."
    )
    parser.add_argument(
        "--n_embd",
        type=int,
        default=384,
        help="Number of hidden units."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate."
    )
    parser.add_argument(
        "--bias",
        action="store_true",
        default=False,
        help="Use bias in mlp."
    )
    parser.add_argument(
        "--max_node_count",
        type=int,
        default=1024,
        help="Restrict training data to graph pairs with under this number of nodes.\
              Set to 0 to train on all data."
    )
    parser.add_argument(
        "--max_nodes_per_batch",
        type=int,
        default=0,
        help="Max nodes in a 'dynamic' batch while training. Set to 0 to disable and use a fixed batch size."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Initial learning rate."
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
        "--hole_scheme",
        type=str,
        default="both",
        choices=("holes", "no_holes", "both"),
        help="Evaluate with or wthout joints whose geometry contains holes."
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=("bce", "mle", "focal"),
        default="mle",
        help="Loss to use."
    )
    parser.add_argument(
        "--loss_sym",
        action="store_true",
        default=False,
        help="Use symmetric loss."
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=5.0,
        help="Gamma parameter in focal loss to down-weight easy examples."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Alpha parameter in focal loss is the weight assigned to rare classes."
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        default=False,
        help="Change wandb to offline mode."
    )
    return parser


def get_args():
    """Get the args used for training"""
    parser = get_parser()
    return args_common.get_args(parser)
