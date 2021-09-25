import logging
import pathlib
import pickle
from collections import defaultdict
from os import listdir
from typing import Union, Dict, AnyStr

import numpy as np

from metadrive.component.blocks.first_block import FirstPGBlock
from metadrive.component.map.argoverse_map import ArgoverseMap
from metadrive.engine.engine_utils import get_global_config, get_engine
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.map_manager import MapManager
from metadrive.utils import is_win, ARGOVERSE_AGENT_ID
from metadrive.utils.random_utils import get_np_random

try:
    from argoverse.map_representation.map_api import ArgoverseMap as AGMap
except ImportError:
    pass

argoverse_city = "PIT"
argoverse_map_center = [2599.5505965123866, 1200.0214763629717]
argoverse_map_radius = 300
argoverse_spawn_lane_index = ('7903', '9713', 0)
argoverse_destination_node = "968"
argoverse_log_id = "c6911883-1843-3727-8eaa-41dc8cda8993"

logging.basicConfig(level=logging.INFO)


class ArgoverseMapManager(MapManager):
    def __init__(self, *args, **kwargs):
        super(ArgoverseMapManager, self).__init__(*args, **kwargs)
        self.ag_map = AGMap()
        self.cached_maps = {}

    def reset(self):
        if self.current_map is None:
            config = self.engine.global_config["map_config"]
            config_key = "%.1f-%.1f" % (config["center"][0], config['center'][1])
            if config_key in self.cached_maps:
                logging.info("use existing maps")
                map = self.cached_maps[config_key]
            else:
                logging.info("use new maps")
                map = self.spawn_object(
                    ArgoverseMap,
                    ag_map=self.ag_map,
                    map_config=config,
                )
                self.cached_maps[config_key] = map
            self.engine.map_manager.load_map(map)
        else:
            print("Already have current map!")


class ArgoverseEnv(MetaDriveEnv):
    def __init__(self, log_id, *args, **kwargs):
        if log_id:
            root_path = pathlib.PurePosixPath(__file__).parent.parent if not is_win(
            ) else pathlib.Path(__file__).resolve().parent.parent
            data_path = root_path.joinpath("assets").joinpath("real_data").joinpath("test_parsed").joinpath(
                "{}.pkl".format(log_id)
            )
            with open(data_path, 'rb') as f:
                loaded_config = pickle.load(f)

            self.map_config = {
                "city": loaded_config["city"],
                "center": loaded_config["map_center"],
                "radius": argoverse_map_radius,
            }
            self.argoverse_config = {
                "agent_pos": {
                    "spawn_lane_index": loaded_config["agent_spawn_lane_index"],
                    "destination_node": loaded_config["agent_targ_node"]
                },
                "locate_info": loaded_config["locate_info"]
            }
        else:
            log_id = argoverse_log_id
            root_path = pathlib.PurePosixPath(__file__).parent.parent if not is_win(
            ) else pathlib.Path(__file__).resolve().parent.parent
            data_path = root_path.joinpath("assets").joinpath("real_data").joinpath("{}.pkl".format(log_id))
            with open(data_path, 'rb') as f:
                locate_info, _ = pickle.load(f)
            self.map_config = {
                "city": argoverse_city,
                "center": argoverse_map_center,
                "radius": argoverse_map_radius,
            }
            self.argoverse_config = {
                "agent_pos": {
                    "spawn_lane_index": argoverse_spawn_lane_index,
                    "destination_node": argoverse_destination_node,
                },
                "locate_info": locate_info
            }
        agent_info = list(self.argoverse_config["locate_info"][ARGOVERSE_AGENT_ID]["traj"].values())
        FREQ = 10
        self.agent_init_speed = (agent_info[1] - agent_info[0]) * FREQ
        self.argoverse_config["locate_info"].pop(ARGOVERSE_AGENT_ID)
        super(ArgoverseEnv, self).__init__(*args, **kwargs)

    def _post_process_config(self, config):
        config = super(ArgoverseEnv, self)._post_process_config(config)
        config["vehicle_config"]["spawn_lane_index"] = self.argoverse_config["agent_pos"]["spawn_lane_index"]
        config["vehicle_config"]["destination_node"] = self.argoverse_config["agent_pos"]["destination_node"]
        config.update({"real_data_config": {"locate_info": self.argoverse_config["locate_info"]}})
        config["traffic_density"] = 0.0  # Remove rule-based traffic flow
        config["map_config"].update(self.map_config)
        return config

    def setup_engine(self):
        super(ArgoverseEnv, self).setup_engine()
        from metadrive.manager.real_data_manager import RealDataManager
        self.engine.register_manager("real_data_manager", RealDataManager())
        self.engine.update_manager("map_manager", ArgoverseMapManager())


class ArgoverseMultiEnv(MetaDriveEnv):
    def __init__(self, config: dict = None):
        self.mode = config.pop("mode")
        super(ArgoverseMultiEnv, self).__init__(config)
        # check parsed training and testing data
        root_path = pathlib.PurePosixPath(__file__).parent.parent if not is_win() else pathlib.Path(__file__).resolve(
        ).parent.parent
        self.file_path = root_path.joinpath("assets").joinpath("real_data").joinpath("{}_parsed".format(self.mode))
        self.data_files = listdir(self.file_path)
        self.envs = [
            ArgoverseEnv(data_file.split(".")[0])
            for data_file in self.data_files[self.start_seed:self.start_seed + self.env_num]
        ]
        self.start_seed = 0

    def reset(self, episode_data: dict = None, force_seed: Union[None, int] = None):
        self._reset_global_seed(force_seed)
        self.current_env = self.envs[self.current_seed]
        self.current_env.reset()

    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        self.current_env.step(actions)


class ArgoverseGeneralizationEnv(MetaDriveEnv):
    def __init__(self, config: dict = None):
        self.mode = config.pop("mode", "train")
        self.source = config.pop("source", "tracking")
        # check parsed training and testing data
        super(ArgoverseGeneralizationEnv, self).__init__(config)
        root_path = pathlib.PurePosixPath(__file__).parent.parent if not is_win() else pathlib.Path(__file__).resolve(
        ).parent.parent
        self.file_path = root_path.joinpath("assets").joinpath("real_data").joinpath("{}_parsed".format(self.mode)) if \
            self.source == 'tracking' else \
            root_path.joinpath("assets").joinpath("real_data").joinpath("{}_forecasting".format(self.mode))
        self.agent_pos_path = root_path.joinpath("assets").joinpath("real_data").joinpath("agent_pos")
        self.data_files = sorted(listdir(self.file_path))

    def reset(self, episode_data: dict = None, force_seed: Union[None, int] = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        setup map and agent parameters
        :param episode_data: Feed the episode data to replay an episode
        :param force_seed: The seed to set the env.
        :return: None
        """
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self._reset_global_seed(force_seed)
        if self.source == 'tracking':
            self._reset_real_config()
        elif self.source == 'forecasting':
            self._reset_real_config_forecasting()
        else:
            assert False
        self.engine.reset()
        if self._top_down_renderer is not None:
            self._top_down_renderer.reset(self.current_map)

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_steps = 0
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)
        assert (len(self.vehicles) == self.num_agents) or (self.num_agents == -1)

        return self._get_reset_return()

    def setup_engine(self):
        super(ArgoverseGeneralizationEnv, self).setup_engine()
        from metadrive.manager.real_data_manager import RealDataManager
        self.engine.register_manager("real_data_manager", RealDataManager())
        self.engine.update_manager("map_manager", ArgoverseMapManager())

    def _reset_real_config_forecasting(self):
        current_data_file = self.data_files[self.current_seed]
        print("map file: ", current_data_file)
        data_path = self.file_path.joinpath(current_data_file)
        with open(data_path, 'rb') as f:
            loaded_config = pickle.load(f)

        locate_info = loaded_config["locate_info"]
        if ARGOVERSE_AGENT_ID not in locate_info.keys():
            agent_id = list(locate_info.keys())[0]
        else:
            agent_id = ARGOVERSE_AGENT_ID

        config = self.engine.global_config
        config["vehicle_config"]["spawn_lane_index"] = locate_info[agent_id]["spawn_lane_index"]
        config["vehicle_config"]["destination_node"] = locate_info[agent_id]["targ_node"]
        config["vehicle_config"].update({"agent_init_pos": locate_info[agent_id]["init_pos"]})
        locate_info.pop(agent_id)

        config.update({"real_data_config": {"locate_info": locate_info, "source": "forecasting"}})
        config["traffic_density"] = 0.0  # Remove rule-based traffic flow
        config["map_config"].update(
            {
                "city": loaded_config["city"],
                "center": ArgoverseMap.metadrive_position(
                    [loaded_config["map_center"][0], -loaded_config["map_center"][1]]
                ),
                "radius": 150
            }
        )

    def _reset_real_config(self):
        try:
            current_data_file = self.data_files[self.current_seed]
        except IndexError:
            print("Index error: {}".format(self.current_seed))
            current_data_file = self.data_files[0]
        current_id = current_data_file.split(".")[0]
        print("map file: ", current_data_file)
        data_path = self.file_path.joinpath(current_data_file)

        with open(data_path, 'rb') as f:
            loaded_config = pickle.load(f)
        map_config = {
            "city": loaded_config["city"],
            "center": loaded_config["map_center"],
            "radius": argoverse_map_radius,
        }

        if current_id in listdir(self.agent_pos_path):
            with open(self.agent_pos_path.joinpath(current_id), 'r') as f:
                spawn_lane_index = eval(f.readline())
                targ_lane_index = eval(f.readline())
            agent_pos = {"spawn_lane_index": spawn_lane_index, "destination_node": targ_lane_index[0]}
        else:
            agent_pos = {
                "spawn_lane_index": loaded_config["agent_spawn_lane_index"],
                "destination_node": loaded_config["agent_targ_node"]
            }

        self.argoverse_config = {"locate_info": loaded_config["locate_info"]}
        agent_init_pos = self.argoverse_config["locate_info"][ARGOVERSE_AGENT_ID]["init_pos"]
        self.argoverse_config["locate_info"].pop(ARGOVERSE_AGENT_ID)

        config = self.engine.global_config
        config["vehicle_config"]["spawn_lane_index"] = agent_pos["spawn_lane_index"]
        config["vehicle_config"]["destination_node"] = agent_pos["destination_node"]
        config["vehicle_config"].update({"agent_init_pos": agent_init_pos})
        config.update({"real_data_config": {"locate_info": self.argoverse_config["locate_info"], "source": "tracking"}})
        config["traffic_density"] = 0.0  # Remove rule-based traffic flow
        config["map_config"].update(
            {
                "city": map_config["city"],
                "center": ArgoverseMap.metadrive_position([map_config["center"][0], -map_config["center"][1]]),
                "radius": map_config["radius"]
            }
        )


def temporary_global_config(config_dict):
    def config_wrapper(func):
        def wrapper(*args, **kwargs):
            origin_config = get_global_config()
            copy_config = origin_config.copy(True)
            origin_config.update(config_dict)
            ret = func(*args, **kwargs)
            get_engine().global_config.update(copy_config)
            return ret

        return wrapper

    return config_wrapper


class ArgoversePGGeneralizationEnv(ArgoverseGeneralizationEnv):
    EXCLUDE_MGR = {"argoverse": ["traffic_manager", "pg_map_manager", "ag_map_manager"],
                   "pg_map": ["real_data_manager", "ag_map_manager", "pg_map_manager"]}

    def setup_engine(self):
        super(ArgoverseGeneralizationEnv, self).setup_engine()
        from metadrive.manager.real_data_manager import RealDataManager
        self.engine.register_manager("real_data_manager", RealDataManager())
        ag_manager = ArgoverseMapManager()
        self.engine.update_manager("map_manager", ag_manager)
        self.engine.register_manager("ag_map_manager", ag_manager)
        self.add_pg_manager()
        self.current_env_type = "argoverse"

    @temporary_global_config({"start_seed": 0, "environment_num": 50})
    def add_pg_manager(self):
        self.engine.register_manager("pg_map_manager", MapManager())

    def reset(self, episode_data: dict = None, force_seed: Union[None, int] = None):
        """
        Reset the env, scene can be restored and replayed by giving episode_data
        Reset the environment or load an episode from episode data to recover is
        setup map and agent parameters
        :param episode_data: Feed the episode data to replay an episode
        :param force_seed: The seed to set the env.
        :return: None
        """
        self.lazy_init()  # it only works the first time when reset() is called to avoid the error when render
        self._reset_global_seed(force_seed)
        if self.source == 'tracking':
            self._reset_real_config()
        elif self.source == 'forecasting':
            self._reset_real_config_forecasting()
        else:
            assert False
        self.random_reset_env()
        if self._top_down_renderer is not None:
            self._top_down_renderer.reset(self.current_map)

        self.dones = {agent_id: False for agent_id in self.vehicles.keys()}
        self.episode_steps = 0
        self.episode_rewards = defaultdict(float)
        self.episode_lengths = defaultdict(int)
        assert (len(self.vehicles) == self.num_agents) or (self.num_agents == -1)

        return self._get_reset_return()

    def random_reset_env(self):
        before_reset_mgr = list(self.engine._managers.keys())
        for manager in self.EXCLUDE_MGR[self.current_env_type]:
            before_reset_mgr.remove(manager)
        if get_np_random().rand() < 0.5:
            new = "pg_map"
            self.engine.update_manager("map_manager", self.engine.pg_map_manager, destroy_old_manager=False)
            self.config["vehicle_config"]["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
            self.config["vehicle_config"]["destination_node"] = None
            # work around
            self.config["traffic_density"] = 0.1
            before_reset_mgr.append("traffic_manager")
            if self.current_env_type == "argoverse":
                self.engine.traffic_manager.density=0.1
        else:
            new = "argoverse"
            self.engine.traffic_manager.density = 0.
            self.engine.update_manager("map_manager", self.engine.ag_map_manager, destroy_old_manager=False)
        self.engine.map_manager.before_reset()
        reset_mgr = list(self.engine._managers.keys())
        for manager in self.EXCLUDE_MGR[new]:
            reset_mgr.remove(manager)

        self.engine.reset(before_reset_managers=before_reset_mgr,
                          reset_managers=reset_mgr,
                          after_reset_managers=reset_mgr)
        self.current_env_type = new


if __name__ == '__main__':
    # env = ArgoverseMultiEnv(dict(mode="train",environment_num=3, start_seed=15, use_render=False))
    # env = ArgoverseGeneralizationEnv(
    #     dict(
    #         mode="all",
    #         source="tracking",
    #         environment_num=74,
    #         start_seed=0,
    #         use_render=True,
    #         manual_control=True,
    #         disable_model_compression=True
    #     )
    # )
    # while True:
    #     env.reset()
    #     env.vehicle.expert_takeover = True
    #     while True:
    #         o, r, d, info = env.step([0., 0.])
    #         info = {}
    #         info["lane_index"] = env.vehicle.lane_index
    #         env.render(text=info)
    #         if d:
    #             break
    #         # print(info)
    #         # print(info)
    for i in range(0, 74):
        print(i)
        env = ArgoversePGGeneralizationEnv(
            dict(mode="all", source="tracking", environment_num=50, start_seed=0, use_render=True, debug=True,
                 traffic_density=0.1,
                 manual_control=True))
        env.reset()
        env.vehicle.expert_takeover = True
        for i in range(1, 20000):
            o, r, d, info = env.step([0., 0.0])
        # except:
        #     print("Error!")
        env.close()
