import argparse
import os
import sys

sys.path.append('/path/to/NeuGrasp')

from pathlib import Path
import rospy


def main(args, round_idx, gpuid, render_frame_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)

    sys.path.append("src")
    from src.neugrasp.main import NeuGraspPlanner

    if args.method == 'neugrasp':
        grasp_planner = NeuGraspPlanner(args)
    else:
        print("No such method!")
        raise NotImplementedError

    if args.rviz:
        rospy.init_node("vgn_vis", anonymous=True)

    from src.gd.experiments import clutter_removal
    clutter_removal.run(
        grasp_plan_fn=grasp_planner,
        logdir=args.logdir,
        description=args.description,
        scene=args.scene,
        object_set=args.object_set,
        num_objects=args.num_objects,
        seed=args.seed,
        sim_gui=args.sim_gui,
        rviz=args.rviz,
        round_idx=round_idx,
        renderer_root_dir=args.renderer_root_dir,
        gpuid=gpuid,
        args=args,
        render_frame_list=render_frame_list
    )


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("---")
            return sys.argv[idx + 1:]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


if __name__ == "__main__":
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]  # get all args after "--"
    round_idx = int(argv[0])
    gpuid = int(argv[1])
    expname = str(argv[2])
    scene = str(argv[3])
    object_set = str(argv[4])
    material_type = str(argv[5])
    blender_asset_dir = str(argv[6])
    log_root_dir = str(argv[7])
    render_frame_list = [int(frame_id) for frame_id in str(argv[8]).replace(' ', '').split(",")]
    method = str(argv[9])
    sim_gui = bool(int(argv[10]))
    rviz = bool(int(argv[11]))
    choose = str(argv[12])

    print("########## Simulation Start ##########")
    print("Round %d\nmethod: %s\nmaterial_type: %s\nviews: %s " % (
        round_idx, method, material_type, str(render_frame_list)))
    print("######################################")

    parser = ArgumentParserForBlender()  ### argparse.ArgumentParser()
    parser.add_argument("---logdir", type=Path, default=expname)
    parser.add_argument("---description", type=str, default="")
    parser.add_argument("---scene", type=str, choices=["pile", "packed", "single"], default=scene)
    parser.add_argument("---object-set", type=str, default=object_set)
    parser.add_argument("---num-objects", type=int, default=5)  # TODO
    parser.add_argument("---seed", type=int, default=42)
    parser.add_argument("---sim-gui", type=bool, default=sim_gui)
    parser.add_argument("---rviz", type=bool, default=rviz)
    parser.add_argument("---choose", type=str, default=choose)

    ###
    parser.add_argument("---renderer_root_dir", type=str, default=blender_asset_dir)
    parser.add_argument("---log_root_dir", type=str, default=log_root_dir)
    parser.add_argument("---obj_texture_image_root_path", type=str, default=blender_asset_dir + "/imagenet")  # NOTE
    parser.add_argument("---cfg_fn", type=str, default="src/neugrasp/configs/neugrasp.yaml")  # NOTE
    parser.add_argument('---database_name', type=str,
                        default='vgn_syn/train/packed/packed_170-220/032cd891d9be4a16be5ea4be9f7eca2b/w_0.8',
                        help='<dataset_name>/<scene_name>/<scene_setting>')

    parser.add_argument("---gen_scene_descriptor", type=bool, default=False)
    parser.add_argument("---load_scene_descriptor", type=bool, default=True)
    parser.add_argument("---material_type", type=str, default=material_type)
    parser.add_argument("---method", type=str, default=method)

    # pybullet camera parameter
    parser.add_argument("---camera_focal", type=float, default=459.14)

    ###
    args = parser.parse_args()
    main(args, round_idx, gpuid, render_frame_list)
