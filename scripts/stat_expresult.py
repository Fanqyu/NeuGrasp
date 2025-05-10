import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append('/path/to/NeuGrasp')
from src.gd import io

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
log_root_dir = str(argv[0])
expname = str(argv[1])
result_flag = int(argv[2])


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")
        self.tsdf_1 = pd.read_csv(logdir / "tsdf_1.csv")
        self.tsdf_05 = pd.read_csv(logdir / "tsdf_05.csv")
        self.tsdf_03 = pd.read_csv(logdir / "tsdf_03.csv")
        self.tsdf_01 = pd.read_csv(logdir / "tsdf_01.csv")
        self.tsdf_005 = pd.read_csv(logdir / "tsdf_005.csv")

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
        # print(df)
        # print(df["cleared_count"].sum())
        # print(df["object_count"].sum())
        # print(self.rounds['object_count'].sum())
        # raise
        # return df["cleared_count"].sum() / df["object_count"].sum() * 100  TODO
        return df["cleared_count"].sum() / self.rounds['object_count'].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):  # TODO
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label

    def l1_1(self):
        return self.tsdf_1["L1"].mean()

    def l1_05(self):
        return self.tsdf_05["L1"].mean()

    def l1_03(self):
        return self.tsdf_03["L1"].mean()

    def l1_01(self):
        return self.tsdf_01["L1"].mean()

    def l1_005(self):
        return self.tsdf_005["L1"].mean()

    def accuracy(self):
        return self.tsdf_1["Acc"].mean()

    def completeness(self):
        return self.tsdf["Comp"].mean()

    def chamfer_distance(self):
        return self.tsdf["Chamfer"].mean()

    def precision(self):
        return self.tsdf["Prec"].mean()

    def recall(self):
        return self.tsdf["Recall"].mean()

    def f_score(self):
        return self.tsdf["F-score"].mean()


##############################
# Combine all trials
##############################
root_path = os.path.join(log_root_dir, "exp_results", expname)

round_dir_list = sorted(os.listdir(root_path))

if not os.path.exists(root_path + "_combine"):
    os.makedirs(root_path + "_combine")

df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "grasps.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "grasps.csv"), index=False)

df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "rounds.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "rounds.csv"), index=False)

df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "tsdf_1.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "tsdf_1.csv"), index=False)

df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "tsdf_05.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "tsdf_05.csv"), index=False)

df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "tsdf_03.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "tsdf_03.csv"), index=False)

df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "tsdf_01.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "tsdf_01.csv"), index=False)

df = pd.DataFrame()
for i in range(len(round_dir_list)):
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "tsdf_005.csv"))
    df_round["round_id"] = i
    df = pd.concat([df, df_round])
df = df.reset_index(drop=True)
df.to_csv(os.path.join(root_path + "_combine", "tsdf_005.csv"), index=False)
##############################
# Print Stat
##############################
logdir = Path(os.path.join(log_root_dir, "exp_results", expname + "_combine"))
data = Data(logdir)

# First, we compute the following metrics for the experiment:
# * **Success rate**: the ratio of successful grasp executions,
# * **Percent cleared**: the percentage of objects removed during each round,
try:
    print("Path:              ", path := str(logdir))
    print("Num grasps:        ", num_grasps := data.num_grasps())
    print("Success rate:      ", SR := data.success_rate())
    print("Percent cleared:   ", DR := data.percent_cleared())
except:
    print("[W] Incomplete results, exit")
    exit()
##############################
# Calc first-time grasping SR
##############################

sum_label = 0
firstgrasp_fail_expidx_list = []
for i in range(len(round_dir_list)):
    # print(i)
    df_round = pd.read_csv(os.path.join(root_path, round_dir_list[i], "grasps.csv"))
    df = df_round.iloc[0:1, :]

    label = df[["label"]].to_numpy(np.float32)
    if label.shape[0] == 0:
        firstgrasp_fail_expidx_list.append(i)
        continue
    sum_label += label[0, 0]
    if label[0, 0] == 0:
        firstgrasp_fail_expidx_list.append(i)

print("First grasp success rate: ", FSR := sum_label / len(round_dir_list))
print("First grasp fail:", len(firstgrasp_fail_expidx_list), "/", len(round_dir_list), ", round id: ",
      firstgrasp_fail_expidx_list)
print("L1_1.0:                  ", l1_1 := str(data.l1_1()))
print("L1_0.5:                  ", l1_05 := str(data.l1_05()))
print("L1_0.3:                  ", l1_03 := str(data.l1_03()))
print("L1_0.1:                  ", l1_01 := str(data.l1_01()))
print("L1_0.05:                  ", l1_005 := str(data.l1_005()))
print("Accuracy:            ", acc := str(data.accuracy()))
# print("Completeness:        ", comp := str(data.completeness()))
# print("Chamfer distance:    ", chamfer := str(data.chamfer_distance()))
# print("Precision:           ", prec := str(data.precision()))
# print("Recall:              ", recall := str(data.recall()))
# print("F-score:             ", f_score := str(data.f_score()))

if result_flag:
    with open(os.path.join(log_root_dir, "exp_results", "summary_results.txt"), 'w') as f:
        f.write("Experiment name:    " + str(expname) + '\n')
        f.write("Num grasps:         " + str(num_grasps) + '\n')
        f.write("Success rate:       " + str(SR) + '\n')
        f.write("Percent cleared:    " + str(DR) + '\n')
        f.write("First success rate: " + str(FSR) + '\n')
        f.write("First grasp fail:   " + str(len(firstgrasp_fail_expidx_list)) + '/' + str(len(round_dir_list)) + '\n')
        f.write("First grasp fail round id:   " + str(firstgrasp_fail_expidx_list) + '\n')
        f.write("L1_1.0:                 " + str(l1_1) + '\n')
        f.write("L1_0.5:                 " + str(l1_05) + '\n')
        f.write("L1_0.3:                 " + str(l1_03) + '\n')
        f.write("L1_0.1:                 " + str(l1_01) + '\n')
        f.write("L1_0.05:                 " + str(l1_005) + '\n')
        f.write("Accuracy:           " + str(acc) + '\n')
        # f.write("Completeness:       " + str(comp) + '\n')
        # f.write("Chamfer distance:   " + str(chamfer) + '\n')
        # f.write("Precision:          " + str(prec) + '\n')
        # f.write("Recall:             " + str(recall) + '\n')
        # f.write("F-score:            " + str(f_score) + '\n')
