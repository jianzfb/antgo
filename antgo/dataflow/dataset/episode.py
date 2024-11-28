from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import gzip
import random
import numpy as np
from antgo.dataflow.dataset.dataset import *
from functools import partial
from h5py import File, Group, Dataset
import numpy as np
import time
import copy

__all__ = ['Episode']


def load_content_from_h5_file(file):
    if isinstance(file, (File, Group)):
        return {key: load_content_from_h5_file(file[key]) for key in list(file.keys())}
    elif isinstance(file, Dataset):
        return file[()]
    else:
        raise NotImplementedError(f"Unspported h5 file type: {type(file)}")


def load_traj_hdf5(path, num_traj=None):
    print('Loading HDF5 file', path)
    file = File(path, 'r')
    keys = list(file.keys())
    if num_traj is not None:
        assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
        keys = sorted(keys, key=lambda x: int(x.split('_')[-1]))
        keys = keys[:num_traj]
    ret = {
        key: load_content_from_h5_file(file[key]) for key in keys
    }
    file.close()
    print('Loaded')
    return ret


TARGET_KEY_TO_SOURCE_KEY = {
    'states': 'env_states',
    'observations': 'obs',
    'success': 'success',
    'next_observations': 'obs',
    # 'dones': 'dones',
    # 'rewards': 'rewards',
    'actions': 'actions',
}


def reorder_keys(d, ref_dict):
    out = dict()
    for k, v in ref_dict.items():
        if isinstance(v, dict):
            out[k] = reorder_keys(d[k], ref_dict[k])
        else:
            out[k] = d[k]
    return out


def load_demo_dataset(path, keys=['observations', 'actions'], num_traj=None, concat=True):
    # assert num_traj is None
    raw_data = load_traj_hdf5(path, num_traj)
    _traj = raw_data['traj_0']
    for key in keys:
        source_key = TARGET_KEY_TO_SOURCE_KEY[key]
        assert source_key in _traj, f"key: {source_key} not in traj_0: {_traj.keys()}"
    dataset = {}
    for target_key in keys:
        # if 'next' in target_key:
        #     raise NotImplementedError('Please carefully deal with the length of trajectory')
        source_key = TARGET_KEY_TO_SOURCE_KEY[target_key]
        dataset[target_key] = [ raw_data[idx][source_key] for idx in raw_data ]
        if isinstance(dataset[target_key][0], np.ndarray) and concat:
            if target_key in ['observations', 'states'] and \
                    len(dataset[target_key][0]) > len(raw_data['traj_0']['actions']):
                dataset[target_key] = np.concatenate([
                    t[:-1] for t in dataset[target_key]
                ], axis=0)
            elif target_key in ['next_observations', 'next_states'] and \
                    len(dataset[target_key][0]) > len(raw_data['traj_0']['actions']):
                dataset[target_key] = np.concatenate([
                    t[1:] for t in dataset[target_key]
                ], axis=0)
            else:
                dataset[target_key] = np.concatenate(dataset[target_key], axis=0)

            print('Load', target_key, dataset[target_key].shape)
        else:
            print('Load', target_key, len(dataset[target_key]), type(dataset[target_key][0]))
    return dataset


def build_state_obs_extractor():
    # env_name = env_id.split("-")[0]
    # if env_name in ["TurnFaucet", "StackCube"]:
    #     return lambda obs: list(obs["extra"].values())
    # elif env_name == "PushChair" or env_name == "PickCube":
    #     return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())
    # else:
    #     raise NotImplementedError(f"Please tune state obs by yourself")
    return lambda obs: list(obs["agent"].values()) + list(obs["extra"].values())


def convert_obs(obs, concat_fn, transpose_fn, state_obs_extractor):
    img_dict = obs["sensor_data"]
    new_img_dict = {
        key: transpose_fn(
            concat_fn([v[key] for v in img_dict.values()])
        )  # (C, H, W) or (B, C, H, W)
        for key in ["rgb", "depth"]
    }
    # if isinstance(new_img_dict['depth'], torch.Tensor): # MS2 vec env uses float16, but gym AsyncVecEnv uses float32
    #     new_img_dict['depth'] = new_img_dict['depth'].to(torch.float16)

    # Unified version
    states_to_stack = state_obs_extractor(obs)
    for j in range(len(states_to_stack)):
        if states_to_stack[j].dtype == np.float64:
            states_to_stack[j] = states_to_stack[j].astype(np.float32)
    try:
        state = np.hstack(states_to_stack)
    except:  # dirty fix for concat trajectory of states
        state = np.column_stack(states_to_stack)
    if state.dtype == np.float64:
        for x in states_to_stack:
            print(x.shape, x.dtype)
        import pdb

        pdb.set_trace()

    out_dict = {
        "state": state,
        "rgb": new_img_dict["rgb"],
        "depth": new_img_dict["depth"],
    }
    return out_dict


obs_process_fn = partial(
    convert_obs,
    concat_fn=partial(np.concatenate, axis=-1),
    transpose_fn=partial(
        np.transpose, axes=(0, 3, 1, 2)
    ),  # (B, H, W, C) -> (B, C, H, W)
    state_obs_extractor=build_state_obs_extractor(),
)

class Episode(Dataset):
    def __init__(self, train_or_test='train', dir=None, **kwargs):
        data_path = kwargs.get('data_path', None)
        obs_horizon = kwargs.get('obs_horizon', 16)
        pred_horizon = kwargs.get('pred_horizon', 8)
        trajectories = load_demo_dataset(data_path, concat=False)
        print("Raw trajectory loaded, start to pre-process the observations...")

        # Pre-process the observations, make them align with the obs returned by the obs_wrapper
        obs_traj_dict_list = []
        ref_dict = {
            'agent':{
                'qpos': '',
                'qvel': ''
            },
            'extra': {
                # 'is_grasped': '',
                'tcp_pose': '',
                # 'goal_pos': ''
            },
            'sensor_param': {
                'base_camera': {
                    'extrinsic_cv': '',
                    'cam2world_gl': '',
                    'intrinsic_cv': ''
                }
            },
            'sensor_data': {
                'base_camera':{
                    'depth': '',
                    'rgb': ''
                }
            }
        }
        for obs_traj_dict in trajectories["observations"]:
            _obs_traj_dict = reorder_keys(
                obs_traj_dict, ref_dict
            )  # key order in demo is different from key order in env obs
            _obs_traj_dict = obs_process_fn(_obs_traj_dict)
            _obs_traj_dict["depth"] = _obs_traj_dict["depth"].astype(np.float32) / 1024
            obs_traj_dict_list.append(_obs_traj_dict)
        trajectories["observations"] = obs_traj_dict_list
        self.obs_keys = list(_obs_traj_dict.keys())
        # Pre-process the actions
        # for i in range(len(trajectories["actions"])):
        #     trajectories["actions"][i] = torch.Tensor(trajectories["actions"][i])
        # print(
        #     "Obs/action pre-processing is done, start to pre-compute the slice indices..."
        # )

        # Pre-compute all possible (traj_idx, start, end) tuples, this is very specific to Diffusion Policy
        # if (
        #     "delta_pos" in args.control_mode
        #     or args.control_mode == "base_pd_joint_vel_arm_pd_joint_vel"
        # ):
        #     self.pad_action_arm = torch.zeros(
        #         (trajectories["actions"][0].shape[1] - 1,)
        #     )
        #     # to make the arm stay still, we pad the action with 0 in 'delta_pos' control mode
        #     # gripper action needs to be copied from the last action
        # else:
        #     raise NotImplementedError(f"Control Mode {args.control_mode} not supported")
        self.pad_action_arm = np.zeros(
            (1, trajectories["actions"][0].shape[1] - 1)
        )
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.slices = []
        num_traj = len(trajectories["actions"])
        total_transitions = 0
        for traj_idx in range(num_traj):
            L = trajectories["actions"][traj_idx].shape[0]
            assert trajectories["observations"][traj_idx]["state"].shape[0] == L + 1
            total_transitions += L

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_before = obs_horizon - 1
            # Pad before the trajectory, so the first action of an episode is in "actions executed"
            # obs_horizon - 1 is the number of "not used actions"
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(-pad_before, L - pred_horizon + pad_after)
            ]  # slice indices follow convention [start, end)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self.trajectories = trajectories

    @property
    def size(self):
        return len(self.slices)

    def at(self, index):
        traj_idx, start, end = self.slices[index]
        L, act_dim = self.trajectories["actions"][traj_idx].shape

        obs_traj = self.trajectories["observations"][traj_idx]
        obs_seq = {}
        for k, v in obs_traj.items():
            obs_seq[k] = v[
                max(0, start) : start + self.obs_horizon
            ]  # start+self.obs_horizon is at least 1
            if start < 0:  # pad before the trajectory
                pad_obs_seq = np.stack([obs_seq[k][0]] * abs(start), axis=0)
                obs_seq[k] = np.concatenate((pad_obs_seq, obs_seq[k]), axis=0)
            # don't need to pad obs after the trajectory, see the above char drawing

        if end < 1:
            gripper_action = self.trajectories["actions"][traj_idx][0:1, 0:1]
            pad_action = np.concatenate((self.pad_action_arm, gripper_action), axis=1)
            act_seq = pad_action.repeat(self.pred_horizon, axis=0)
        else:
            act_seq = self.trajectories["actions"][traj_idx][max(0, start) : end]
            if start < 0:  # pad before the trajectory
                act_seq = np.concatenate([act_seq[0:1].repeat(-start, axis=0), act_seq], axis=0)
            if end > L:  # pad after the trajectory
                gripper_action = act_seq[-1:, -1:]  # assume gripper is with pos controller
                pad_action = np.concatenate((self.pad_action_arm, gripper_action), axis=1)
                act_seq = np.concatenate([act_seq, pad_action.repeat(end - L, axis=0)], axis=0)
                # making the robot (arm and gripper) stay still
        assert (
            obs_seq["state"].shape[0] == self.obs_horizon
            and act_seq.shape[0] == self.pred_horizon
        )
        return {
            "rgb": obs_seq['rgb'],
            "depth": obs_seq['depth'],
            "state": obs_seq['state'],
            "actions": act_seq,
        }

# aa = Episode(data_path='/workspace/project/ManiSkill/bbb/jian.h5')
# print(aa.size)
# for idx in range(200):
#     aa.at(idx)
#     print(idx)