import pathlib
import pickle

from metadrive.component.map.argoverse_map import ArgoverseMap
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.map_manager import MapManager
from metadrive.utils import is_win, ARGOVERSE_AGENT_ID

argoverse_city = "PIT"
argoverse_map_center = [2599.5505965123866, 1200.0214763629717]
argoverse_map_radius = 300
argoverse_spawn_lane_index = ('7903', '9713', 0)
argoverse_destination_node = "968"
argoverse_log_id = "c6911883-1843-3727-8eaa-41dc8cda8993"


class ArgoverseMapManager(MapManager):

    def __init__(self, map_config, *args, **kwargs):
        super(ArgoverseMapManager, self).__init__(*args, **kwargs)
        self.map_config = map_config
    
    def before_reset(self):
        # do not unload map
        pass

    def reset(self):
        if self.current_map is None:
            self.engine.global_config["map_config"].update(
                {
                    "city": self.map_config["city"],
                    "center": ArgoverseMap.metadrive_position([self.map_config["center"][0], -self.map_config["center"][1]]),
                    "radius": self.map_config["radius"]
                }
            )
            map = ArgoverseMap(self.engine.global_config["map_config"])
            self.engine.map_manager.load_map(map)


class ArgoverseEnv(MetaDriveEnv):

    def __init__(self, log_id, *args, **kwargs):
        if log_id:
            root_path = pathlib.PurePosixPath(__file__).parent.parent if not is_win() else pathlib.Path(__file__).resolve(
            ).parent.parent
            data_path = root_path.joinpath("assets").joinpath("real_data").joinpath("test_parsed").joinpath("{}.pkl".format(log_id))
            with open(data_path, 'rb') as f:
                loaded_config = pickle.load(f)
                
            self.argoverse_config = {
                "map_config": {
                    "city": loaded_config["city"],
                    "center": loaded_config["map_center"],
                    "radius": argoverse_map_radius,
                },
                "agent_pos": {
                    "spawn_lane_index": loaded_config["agent_spawn_lane_index"],
                    "destination_node": loaded_config["agent_targ_node"]
                },
                "locate_info": loaded_config["locate_info"]
            }
        else:
            log_id = argoverse_log_id
            root_path = pathlib.PurePosixPath(__file__).parent.parent if not is_win() else pathlib.Path(__file__).resolve(
            ).parent.parent
            data_path = root_path.joinpath("assets").joinpath("real_data").joinpath("{}.pkl".format(log_id))
            with open(data_path, 'rb') as f:
                locate_info, _ = pickle.load(f)
            self.argoverse_config = {
                "map_config": {
                    "city": argoverse_city,
                    "center": argoverse_map_center,
                    "radius": argoverse_map_radius,
                },
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
        config["vehicle_config"].update({"init_speed": self.agent_init_speed})
        config.update({"real_data_config": {"locate_info": self.argoverse_config["locate_info"]}})
        config["traffic_density"] = 0.0  # Remove rule-based traffic flow
        return config

    def setup_engine(self):
        super(ArgoverseEnv, self).setup_engine()
        from metadrive.manager.real_data_manager import RealDataManager
        self.engine.register_manager("real_data_manager", RealDataManager())
        self.engine.update_manager("map_manager", ArgoverseMapManager(self.argoverse_config["map_config"]))
