# 这仅仅是一个测试


# %%
import os
import math
import torch
import torch.utils.data
import numpy as np
import json


def read_from_json(target_dir):
    f = open(target_dir, "r")
    data = json.load(f)
    data = json.loads(data)
    # print(data)
    return data


# %%
if __name__ == "__main__":
    pose_dict = read_from_json(
        "E:\\设计大赛\\Music-Dance-Video-Synthesis-master\\dataset\\ballet_revised_pose_pairs.json")
    length = 0
    keys = sorted(pose_dict.keys())
    for key in keys:
        #index = str("%03d" % i)
        sub_keys = sorted(pose_dict[str(key)].keys())
        if key == "046":
            break
        for sub_key in sub_keys:
            temp_pose = np.array(
                pose_dict[str(key)][str(sub_key)]["joint_coors"])
            if(temp_pose.shape == (100,)):
                print("girl"+key+" "+sub_key+" is wrong")
                continue
            length += 1

    target = torch.FloatTensor(2*length, 50, 1600).zero_()
    label = torch.FloatTensor(2*length, 50, 18, 2).zero_()
    index = 0

    keys = sorted(pose_dict.keys())
    # keys=["017","018"]
    for key in keys:
        #index = str("%03d" % i)
        sub_keys = sorted(pose_dict[str(key)].keys())
        if key == "046":
            break
        for sub_key in sub_keys:

            print(key+" "+sub_key)
            temp_audio = np.array(
                pose_dict[str(key)][str(sub_key)]['audio_sequence'])

            temp_pose = np.array(
                pose_dict[str(key)][str(sub_key)]["joint_coors"])
            if(temp_pose.shape == (100,)):
                continue
            x_coor = (temp_pose[:, :, 0]/320)-1
            y_coor = (temp_pose[:, :, 1]/180)-1
            temp = np.zeros((100, 18, 2))
            temp[:, :, 0] = x_coor
            temp[:, :, 1] = y_coor
            temp_pose = temp

            d = torch.from_numpy(temp_audio).type(torch.LongTensor)

            slices1 = d[0:80000].view(50, 1600)
            slices2 = d[80000:160000].view(50, 1600)
            target[index] = slices1
            target[index+1] = slices2

            label[index] = torch.from_numpy(temp_pose[0:50, :, :])
            label[index+1] = torch.from_numpy(temp_pose[50:100, :, :])
            index += 2
# %%
