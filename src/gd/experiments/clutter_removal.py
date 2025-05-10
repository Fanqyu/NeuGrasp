import collections
import os
import sys
import uuid

import mcubes
import open3d as o3d
import pandas as pd
from colorama import Fore

sys.path.append("/path/to/NeuGrasp")
import shutil
from pathlib import Path
from src.gd import io, vis
from src.gd.grasp import *
from src.gd.simulation import ClutterRemovalSim
from src.gd.utils.transform import Transform
from src.rd.render import blender_init_scene, blender_render, blender_update_sceneobj


MAX_CONSECUTIVE_FAILURES = 2

State = collections.namedtuple("State", ["tsdf", "pc"])


def copydirs(from_file, to_file):
    if not os.path.exists(to_file):
        os.makedirs(to_file)
    files = os.listdir(from_file)
    for f in files:
        if os.path.isdir(from_file + '/' + f):
            copydirs(from_file + '/' + f, to_file + '/' + f)
        else:
            shutil.copy(from_file + '/' + f, to_file + '/' + f)


def run(
        grasp_plan_fn,
        logdir,
        description,
        scene,
        object_set,
        num_objects=5,
        n=6,
        N=None,
        seed=1,
        sim_gui=False,
        rviz=False,
        round_idx=0,
        renderer_root_dir="",
        gpuid=None,
        args=None,
        render_frame_list=[],
        scene_o3d_vis=False
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, renderer_root_dir=renderer_root_dir, args=args)
    logger = Logger(args.log_root_dir, logdir, description, round_idx)

    output_modality_dict = {'RGB': 1,
                            'IR': 0,
                            'NOCS': 0,
                            'Mask': 0,
                            'Normal': 0}

    for n_round in range(round_idx, round_idx + 1):
        urdfs_and_poses_dict = sim.reset(num_objects, round_idx)

        renderer, quaternion_list, translation_list, path_scene = blender_init_scene(renderer_root_dir,
                                                                                     args.log_root_dir,
                                                                                     args.obj_texture_image_root_path,
                                                                                     scene, urdfs_and_poses_dict,
                                                                                     round_idx, logdir,
                                                                                     False,
                                                                                     args.material_type, gpuid,
                                                                                     output_modality_dict)

        render_finished = False
        render_fail_times = 0
        while not render_finished and render_fail_times < 3:
            try:
                blender_render(renderer, quaternion_list, translation_list, path_scene, render_frame_list,
                               output_modality_dict, args.camera_focal, is_init=True)
                render_finished = True
            except:
                render_fail_times += 1
        if not render_finished:
            raise RuntimeError("Blender render failed for 3 times.")

        path_scene_backup = os.path.join(path_scene + "_backup", "%d" % n_round)
        if not os.path.exists(path_scene_backup):
            os.makedirs(path_scene_backup)
        copydirs(path_scene, path_scene_backup)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)

        consecutive_failures = 1
        last_label = None

        n_grasp = 0
        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            pre_tsdf_ori = None

            timings = {}

            timings["integration"] = 0

            gt_tsdf, gt_tsdf_hr, _ = sim.acquire_tsdf(n=n, N=N)  # tsdf in here ranges from 0 to 1
            pc = gt_tsdf_hr.get_cloud()

            if scene_o3d_vis:
                o3d.visualization.draw_geometries([gt_tsdf.get_cloud()])
                o3d.visualization.draw_geometries([gt_tsdf_hr.get_cloud()])

            if rviz:
                vis.clear()
                vis.draw_workspace(sim.size)
                vis.draw_points(np.asarray(pc.points))

            if args.method == "neugrasp":
                grasps, scores, timings["planning"], pre_tsdf = grasp_plan_fn(render_frame_list, round_idx, n_grasp,
                                                                              gt_tsdf)
            else:
                raise NotImplementedError

            if len(grasps) == 0:
                print(f"[I] {Fore.YELLOW}No detections found, abort this round{Fore.RESET}")
                break
            else:
                print(f"[I] {Fore.GREEN}{len(grasps)}{Fore.RESET} detections are found")

            # execute grasp
            grasp, score = grasps[0], scores[0]  # select first from an ordered permutation

            if rviz:
                pre_tsdf = pre_tsdf.squeeze()
                gt_tsdf_grid = gt_tsdf.get_grid().squeeze()
                invalid_mask = gt_tsdf_grid == 0.
                pre_tsdf[invalid_mask] = -1
                vis.draw_tsdf((pre_tsdf + 1) / 2., gt_tsdf.voxel_size)
                # vis.draw_tsdf(gt_tsdf_grid.squeeze(), gt_tsdf.voxel_size)
                vis.draw_grasp(grasp, score, 0.05)
                vis.draw_grasps(grasps, scores, 0.05)
                if input(
                    f'If you\'d like to execute, input `{Fore.GREEN}y{Fore.RESET}` and press `{Fore.GREEN}Enter{Fore.RESET}`: ') == 'y':
                    (label, _), remain_obj_inws_infos = sim.execute_grasp(grasp, allow_contact=True)
                else:
                    print(f"{Fore.RED}Terminating...{Fore.RED}")
                    sys.exit(1)
            else:
                (label, _), remain_obj_inws_infos = sim.execute_grasp(grasp, allow_contact=True)

            # render the modified scene after grasping
            obj_name_list = [str(value[0]).split("/")[-1][:-5] for value in remain_obj_inws_infos]
            obj_pose_list = [Transform.from_matrix(value[2]) for value in remain_obj_inws_infos]
            obj_quat_list = [pose.rotation.as_quat()[[3, 0, 1, 2]] for pose in obj_pose_list]
            obj_trans_list = [pose.translation for pose in obj_pose_list]
            obj_uid_list = [value[3] for value in remain_obj_inws_infos]

            # update blender scene
            blender_update_sceneobj(obj_name_list, obj_trans_list, obj_quat_list, obj_uid_list)

            # render updated scene
            render_finished = False
            render_fail_times = 0
            while not render_finished and render_fail_times < 3:
                try:
                    blender_render(renderer, quaternion_list, translation_list, path_scene, render_frame_list,
                                   output_modality_dict, args.camera_focal)
                    render_finished = True
                except:
                    render_fail_times += 1
            if not render_finished:
                raise RuntimeError("Blender render failed for 3 times.")

            path_scene_backup = os.path.join(path_scene + "_backup", "%d_%d" % (n_round, n_grasp))
            if not os.path.exists(path_scene_backup):
                os.makedirs(path_scene_backup)
            copydirs(path_scene, path_scene_backup)

            # log the grasp
            logger.log_grasp(round_id, timings, grasp, score, label)
            logger.log_tsdf(round_id, gt_tsdf, pre_tsdf, n_grasp, pc,
                            pre_tsdf_ori)  # NOTE: gt_tsdf: class TSDFVolume

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label

            n_grasp += 1


class Logger(object):
    def __init__(self, log_root_dir, expname, description, round_idx):
        self.logdir = Path(os.path.join(log_root_dir, "exp_results", expname, "%04d" % int(round_idx)))  # description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self.tsdf1_csv_path = self.logdir / "tsdf_1.csv"
        self.tsdf005_csv_path = self.logdir / "tsdf_005.csv"
        self.tsdf01_csv_path = self.logdir / "tsdf_01.csv"
        self.tsdf03_csv_path = self.logdir / "tsdf_03.csv"
        self.tsdf05_csv_path = self.logdir / "tsdf_05.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

        if not self.tsdf1_csv_path.exists():
            columns = ['round_id', 'scene_id', 'L1', 'Acc']
            io.create_csv(self.tsdf1_csv_path, columns)
        if not self.tsdf005_csv_path.exists():
            columns = ['round_id', 'scene_id', 'L1', 'Acc']
            io.create_csv(self.tsdf005_csv_path, columns)
        if not self.tsdf01_csv_path.exists():
            columns = ['round_id', 'scene_id', 'L1', 'Acc']
            io.create_csv(self.tsdf01_csv_path, columns)
        if not self.tsdf03_csv_path.exists():
            columns = ['round_id', 'scene_id', 'L1', 'Acc']
            io.create_csv(self.tsdf03_csv_path, columns)
        if not self.tsdf05_csv_path.exists():
            columns = ['round_id', 'scene_id', 'L1', 'Acc']
            io.create_csv(self.tsdf05_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, timings, grasp, score, label):
        # log scene
        scene_id = uuid.uuid4().hex

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )

    def log_tsdf(self, round_id, gt, pre, n_grasp, pc_hr, tsdf_ori=None, nerf=None, downsample=1, threshold=0.25):
        """
        The unit of results is 1 voxel. For instance, 0.5 means 0.5 voxelsizes, which is equal to 0.5 * 0.3 / 40 meters.
        """
        scene_id = uuid.uuid4().hex
        tsdf_gt, pc_gt = gt.get_grid() * 2. - 1., gt.get_cloud()
        tsdf_gt, tsdf_pre = tsdf_gt.squeeze(), pre.squeeze()

        if tsdf_ori is None:
            np.savez_compressed(self.scenes_dir / f'scene_tsdf_{n_grasp}.npz', gt=tsdf_gt, pre=tsdf_pre)
        else:
            np.savez_compressed(self.scenes_dir / f'scene_tsdf_{n_grasp}.npz', gt=tsdf_gt, pre=tsdf_pre,
                                pre_ori=tsdf_ori.squeeze())

        valid_1 = np.logical_and(tsdf_gt > -1, tsdf_gt < 0.1)
        valid_005 = np.logical_and(tsdf_gt > -0.05, tsdf_gt < 0.1)
        valid_01 = np.logical_and(tsdf_gt > -0.1, tsdf_gt < 0.1)
        valid_03 = np.logical_and(tsdf_gt > -0.3, tsdf_gt < 0.1)
        valid_05 = np.logical_and(tsdf_gt > -0.5, tsdf_gt < 0.1)
        l1_surface1 = compute_mae(tsdf_pre, tsdf_gt, valid_1)
        l1_surface005 = compute_mae(tsdf_pre, tsdf_gt, valid_005)
        l1_surface01 = compute_mae(tsdf_pre, tsdf_gt, valid_01)
        l1_surface03 = compute_mae(tsdf_pre, tsdf_gt, valid_03)
        l1_surface05 = compute_mae(tsdf_pre, tsdf_gt, valid_05)

        vertices_pr, _ = mcubes.marching_cubes(tsdf_pre, 0.)
        pc_hr_np = (np.asarray(pc_hr.points) / (0.3 / 120) - 0.5) * 40 / 120
        vertices_hr = o3d.utility.Vector3dVector(pc_hr_np)
        pc_hr = o3d.geometry.PointCloud(vertices_hr)
        v_pr = o3d.utility.Vector3dVector(vertices_pr)
        pc_pr = o3d.geometry.PointCloud(v_pr)

        o3d.io.write_point_cloud(str(self.scenes_dir / f'scene_pc_gt_{n_grasp}.ply'), pc_gt)
        o3d.io.write_point_cloud(str(self.scenes_dir / f'scene_pc_pr_{n_grasp}.ply'), pc_pr)

        dist_gt2pr = pc_hr.compute_point_cloud_distance(pc_pr)
        dist_pr2gt = pc_pr.compute_point_cloud_distance(pc_hr)
        dist_gt2pr, dist_pr2gt = np.asarray(dist_gt2pr), np.asarray(dist_pr2gt)
        acc = np.mean(dist_pr2gt)

        io.append_csv(
            self.tsdf1_csv_path,
            round_id,
            scene_id,
            l1_surface1,
            acc
        )
        io.append_csv(
            self.tsdf01_csv_path,
            round_id,
            scene_id,
            l1_surface01,
            acc
        )
        io.append_csv(
            self.tsdf03_csv_path,
            round_id,
            scene_id,
            l1_surface03,
            acc
        )
        io.append_csv(
            self.tsdf05_csv_path,
            round_id,
            scene_id,
            l1_surface05,
            acc
        )
        io.append_csv(
            self.tsdf005_csv_path,
            round_id,
            scene_id,
            l1_surface005,
            acc
        )


def compute_mae(pr, gt, mask):
    return np.mean(np.abs(pr[mask] - gt[mask]))


class Data(object):
    """
    Object for loading and analyzing experimental data.
    """

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):  # TODO
        scene_id, gripper_type, scale, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label
