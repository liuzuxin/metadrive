from metadrive.component.map.pg_map import PGMap
from metadrive.manager.map_manager import MapManager
from metadrive.envs.alter_param_env import EnvParams


class MapParamManager(MapManager):
    """
    MapManager contains a list of maps
    """
    PRIORITY = 0  # Map update has the most high priority

    def __init__(self, env_params):
        super(MapParamManager, self).__init__()
        self.env_params = env_params

    def add_random_to_map(self, map_config):
        super(MapParamManager, self).add_random_to_map(map_config)
        map_config[PGMap.LANE_WIDTH] = self.env_params.pop(EnvParams.LANE_WIDTH, map_config[PGMap.LANE_WIDTH])
        map_config[PGMap.LANE_NUM] = self.env_params.pop(EnvParams.LANE_NUM, map_config[PGMap.LANE_NUM])
        map_config[PGMap.GENERATE_CONFIG] = self.env_params.pop(EnvParams.BLOCK_NUM, map_config[PGMap.GENERATE_CONFIG])
        return map_config
