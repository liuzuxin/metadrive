from metadrive.manager.traffic_manager import TrafficManager
from metadrive.envs.alter_param_env import EnvParams


class TrafficParamManager(TrafficManager):
    VEHICLE_GAP = 10  # m

    def __init__(self, env_params):
        """
        Control the whole traffic flow
        """
        super(TrafficParamManager, self).__init__()

        self.env_params = env_params
        self.density = self.env_params.pop(EnvParams.DENSITY, self.engine.global_config["traffic_density"])