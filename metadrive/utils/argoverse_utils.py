from argoverse.data_loading.frame_label_accumulator import PerFrameLabelAccumulator
from argoverse.map_representation.map_api import ArgoverseMap as AGMap
import pandas as pd
import numpy as np
from metadrive.constants import ARGOVERSE_AGENT_ID
import os
import pickle

engine_init = False
ag_map = AGMap()


def round_int(num, precise=8):
    return round(num / 10**precise) * 10**precise


def format_pose_files(dataset_dir, timesteps=None):
    data_path = os.path.join(dataset_dir, "poses")
    pose_prefix = "city_SE3_egovehicle_"
    pose_timesteps = sorted([int(i.split("_")[-1].split(".")[0]) for i in os.listdir(data_path)])
    i = 0  # timestep index
    j = 0  # pose timesteps index
    while True:
        if i >= len(timesteps):
            break
        if j >= len(pose_timesteps):
            print("format not complete with {}/{} formated!".format(i, len(timesteps)))
            break
        if round_int(timesteps[i]) == round_int(pose_timesteps[j]):
            f = os.popen(
                "cp -v {}{}.json {}{}.json".format(
                    os.path.join(data_path, pose_prefix), pose_timesteps[j], os.path.join(data_path, pose_prefix),
                    timesteps[i]
                )
            )
            # print(f.readlines())
            f.close()
            i += 1
        j += 1



def _propose_destination(spawn_lane_index, map, traj_lane_num=5):
    argo_lanes = map.blocks[0].argo_lanes
    for lane_key in argo_lanes.keys():
        argo_lane = argo_lanes[lane_key]
        if (argo_lane.start_node, argo_lane.end_node, 0) == spawn_lane_index:
            break

    for _ in range(traj_lane_num - 1):
        if argo_lane.successors is None or len(argo_lane.successors) == 0:
            break
        argo_lane = argo_lanes[np.random.choice(argo_lane.successors)]

    return argo_lane.end_node

def parse_tracking_data(dataset_dir, log_id):

    pfa = PerFrameLabelAccumulator(dataset_dir, dataset_dir, "")
    pfa.accumulate_per_log_data(log_id=log_id)
    log_egopose_dict = pfa.log_egopose_dict[log_id]
    log_timestamp_dict = pfa.log_timestamp_dict[log_id]
    timesteps = sorted(log_egopose_dict.keys())

    self_id = ARGOVERSE_AGENT_ID
    locate_info = {}
    locate_info[self_id] = {
        "init_pos": None,
        "traj": {},
        "heading": None,
    }
    for timestep_index, timestep in enumerate(timesteps):
        xcenter = log_egopose_dict[timestep]['translation'][0]
        ycenter = log_egopose_dict[timestep]['translation'][1]
        if locate_info[self_id]["init_pos"] is None:
            locate_info[self_id]["init_pos"] = np.array([xcenter, -ycenter])
            locate_info[self_id]["diag_len"] = 100
        else:
            locate_info[self_id]["traj"][str(timestep_index)] = np.array([xcenter, -ycenter])

        for i, frame_rec in enumerate(log_timestamp_dict[timestep]):
            bbox_city_fr = frame_rec.bbox_city_fr
            uuid = frame_rec.track_uuid
            center_point = np.mean([bbox_city_fr[0, :2], bbox_city_fr[-1, :2]], axis=0)
            center_point *= np.array([1, -1])
            if uuid not in list(locate_info.keys()):
                locate_info[uuid] = {
                    "init_pos": center_point,
                    "diag_len": np.linalg.norm(bbox_city_fr[0, :2] - bbox_city_fr[-1, :2]),
                    "traj": {},
                    "heading": {
                        str(timestep_index): bbox_city_fr[0, :2] - bbox_city_fr[2, :2]
                    }
                }
                continue
            locate_info[uuid]["traj"][str(timestep_index)] = center_point
            locate_info[uuid]["heading"][str(timestep_index)] = bbox_city_fr[0, :2] - bbox_city_fr[2, :2]

    moving_obj_threshold = 0
    for key in list(locate_info.keys()):
        traj = locate_info[key]["traj"]
        # print(locate_info[key]["diag_len"])
        if len(traj.keys()) == 0:
            locate_info.pop(key)
            continue
        # Remove static objects
        # min_key = min(traj.keys())
        # max_key = max(traj.keys())
        # dist = np.linalg.norm(traj[min_key]-traj[max_key])
        # print(dist)
        # crit1 = np.linalg.norm(traj[min_key]-traj[max_key]) < moving_obj_threshold
        # # Remove objects that reappears
        # # crit2 = int((info['end_t'] - info['start_t']) / 1e8) != len(info['traj'])
        # print(locate_info[key]["diag_len"])
        crit3 = locate_info[key]["diag_len"] < 3
        # # if crit1 or crit2:
        if crit3:
            locate_info.pop(key)

    # ===============get map params=========
    with open(os.path.join(dataset_dir, log_id, "city_info.json"), 'r') as f:
        city_info = eval(f.readline())
    city = city_info["city_name"]
    if ARGOVERSE_AGENT_ID not in locate_info.keys():
        return None
    agent_init_pos = locate_info[ARGOVERSE_AGENT_ID]["init_pos"]
    agent_timesteps = sorted(int(i) for i in locate_info[ARGOVERSE_AGENT_ID]["traj"].keys())
    agent_targ_pos = locate_info[ARGOVERSE_AGENT_ID]["traj"][str(agent_timesteps[-1])]
    # print(agent_init_pos, agent_targ_pos)
    map_center = (agent_init_pos + agent_targ_pos) / 2 * np.array([1, -1])
    # ===============get agent locate info========
    from metadrive.component.map.argoverse_map import ArgoverseMap
    from metadrive.engine.engine_utils import initialize_engine
    from metadrive.envs.metadrive_env import MetaDriveEnv

    global engine_init
    if not engine_init:
        default_config = MetaDriveEnv.default_config()
        engine = initialize_engine(default_config)
        engine_init = True
    config = {
        "city": city,
        # "draw_map_resolution": 1024,
        "center": ArgoverseMap.metadrive_position(map_center),
        "radius": 300
    }
    ag_map = AGMap()
    map = ArgoverseMap(ag_map=ag_map, map_config=config)

    def get_nearest_lane(pos):
        min_dist = 1e6
        min_lane = None
        for lane in map.blocks[0].argo_lanes.values():
            lat, long = lane.local_coordinates(pos)
            if abs(lat) + abs(long) < min_dist:
                min_lane = lane
                min_dist = abs(lat) + abs(long)
        return None if not min_lane else min_lane

    spawn_lane = get_nearest_lane(agent_init_pos)
    spawn_lane_index = spawn_lane.index if spawn_lane else None
    targ_lane = get_nearest_lane(agent_targ_pos)
    # print(targ_lane.segment_property)
    targ_node = targ_lane.start_node
    print(spawn_lane_index, targ_node)
    print(map.road_network.get_lane(spawn_lane_index))

    return {
        "locate_info": locate_info,
        "city": city,
        "map_center": map_center,
        "agent_spawn_lane_index": spawn_lane_index,
        "agent_targ_node": targ_node
    }


def parse_forcasting_data(data_path):
    data = np.array(pd.read_csv(data_path))
    locate_info = {}
    timestep = 0
    for entry in data:
        _, v_id, _, x, y, city = entry
        if v_id not in locate_info.keys():
            locate_info[v_id] = {"init_pos": np.array([x, -y]), "targ_pos": np.array([x, -y])}
        else:
            locate_info[v_id]["targ_pos"] = np.array([x, -y])
        if v_id == ARGOVERSE_AGENT_ID:
            timestep += 1

    agent_init_pos = locate_info[ARGOVERSE_AGENT_ID]["init_pos"]
    agent_targ_pos = locate_info[ARGOVERSE_AGENT_ID]["targ_pos"]
    moving_obj_threshold = 2
    for key in list(locate_info.keys()):
        init_pos = locate_info[key]["init_pos"]
        targ_pos = locate_info[key]["targ_pos"]
        crit1 = np.linalg.norm(init_pos - targ_pos) < moving_obj_threshold
        if crit1 and key is not ARGOVERSE_AGENT_ID:
            locate_info.pop(key)

    # ===============get map locate info========
    from metadrive.component.map.argoverse_map import ArgoverseMap
    from metadrive.engine.engine_utils import initialize_engine
    from metadrive.envs.metadrive_env import MetaDriveEnv

    global engine_init
    if not engine_init:
        default_config = MetaDriveEnv.default_config()
        engine = initialize_engine(default_config)
        engine_init = True

    map_center = (agent_init_pos + agent_targ_pos) / 2 * np.array([1, -1])
    config = {
        "city": city,
        # "draw_map_resolution": 1024,
        "center": ArgoverseMap.metadrive_position(map_center),
        "radius": 200
    }
    map = ArgoverseMap(ag_map=ag_map, map_config=config)

    all_argo_lanes = list(map.blocks[0].argo_lanes.values())
    def get_nearest_lane(pos):
        min_dist = 1e6
        min_lane = None
        for lane in all_argo_lanes:
            lat, long = lane.local_coordinates(pos)
            if abs(lat) + abs(long) < min_dist:
                min_lane = lane
                min_dist = abs(lat) + abs(long)
        all_argo_lanes.remove(min_lane)
        return None if not min_lane else min_lane

    for key in list(locate_info.keys()):
        init_pos = locate_info[key]["init_pos"]
        targ_pos = locate_info[key]["targ_pos"]
        spawn_lane = get_nearest_lane(init_pos)
        spawn_lane_index = (spawn_lane.start_node, spawn_lane.end_node, 0) if spawn_lane else None
        try:
            targ_node = _propose_destination(spawn_lane_index, map)
        except KeyError:
            targ_node = None
        locate_info[key]["spawn_lane_index"] = spawn_lane_index
        locate_info[key]["targ_node"] = targ_node
    if len(locate_info.keys()) == 0:
        return None

    map.destroy()

    return {"locate_info": locate_info, "city": city, "map_center": map_center}


if __name__ == '__main__':
    file_path = "/home/xuezhenghai/argoverse-api/argoverse-forecasting/train"
    output_path = "/home/xuezhenghai/argoverse-api/argoverse-forecasting/train_parsed"
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    for log in os.listdir(file_path):
        print("Parsing log {}".format(log))
        locate_info = parse_forcasting_data(os.path.join(file_path, log))
        with open(os.path.join(output_path, "{}.pkl".format(log.split(".")[0])), 'wb') as f:
            pickle.dump(locate_info, f)

    # file_path = "/home/xuezhenghai/argoverse-api/argoverse-tracking/val/"
    # output_path = "/home/xuezhenghai/argoverse-api/argoverse-tracking/test_parsed"
    # if not os.path.isdir(output_path):
    # os.mkdir(output_path)
    # for log in os.listdir(file_path):
    # print("Parsing log {}".format(log))
    # locate_info = parse_tracking_data(file_path, log)
    # with open(os.path.join(output_path, "{}.pkl".format(log)), 'wb') as f:
    # pickle.dump(locate_info, f)
