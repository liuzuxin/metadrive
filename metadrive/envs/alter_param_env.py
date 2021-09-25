from metadrive.envs.metadrive_env import MetaDriveEnv
import argparse
from metadrive.constants import HELP_MESSAGE
import random
from metadrive.component.map.pg_map import PGMap

import numpy as np


class EnvParams():
    # Available environment parameters
    # ========map param========
    LANE_WIDTH = PGMap.LANE_WIDTH  # ok
    LANE_NUM = PGMap.LANE_NUM  # ok
    BLOCK_NUM = PGMap.GENERATE_CONFIG  # ok
    BLOCK_PROB = "_block_prob"  # TODO

    # ========traffic param=====
    DENSITY = "traffic_density"  # ok
    MAX_SPEED = "_max_speed"  # TODO


class AlterParamEnv(MetaDriveEnv):
    def _post_process_config(self, config):
        config = super(AlterParamEnv, self)._post_process_config(config)
        config["random_lane_width"] = False
        config["random_lane_num"] = False

        return config

    def reset_env_params(self, env_params: dict = None):
        config = self.config
        map_config = config["map_config"]
        for key in list(env_params.keys()):
            if key in map_config.keys():
                map_config.update({key: env_params[key]})
                env_params.pop(key)
            elif key in config:
                config.update({key: env_params[key]})
                env_params.pop(key)


if __name__ == '__main__':
    config = dict(
        # controller="joystick",
        use_render=True,
        manual_control=True,
        traffic_density=0.1,
        environment_num=100,
        random_agent_model=True,
        random_lane_width=True,
        random_lane_num=True,
        map=4,  # seven block
        start_seed=random.randint(0, 1000)
    )
    # Specify environment parameters
    # config.update({EnvParams.LANE_WIDTH: 10, EnvParams.LANE_NUM: 10, EnvParams.BLOCK_NUM: 10, EnvParams.DENSITY: 0.9})
    env = AlterParamEnv(config)
    print(HELP_MESSAGE)
    while True:
        env.reset_env_params(
            {
                EnvParams.LANE_WIDTH: np.random.randint(1, 4),
                EnvParams.LANE_NUM: np.random.randint(1, 4),
                EnvParams.BLOCK_NUM: 10,
                EnvParams.DENSITY: np.random.random()
            }
        )
        o = env.reset()
        env.current_track_vehicle.expert_takeover = True
        for i in range(1, 50):
            o, r, d, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
    env.close()
