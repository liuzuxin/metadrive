from metadrive.envs.metadrive_env import MetaDriveEnv
import argparse
from metadrive.constants import HELP_MESSAGE
import random

import numpy as np

class EnvParams():
    # ========map param========
    LANE_WIDTH = "_lane_width"             # ok
    LANE_NUM = "_lane_num"                 # ok
    BLOCK_NUM = "_block_num"               # ok
    BLOCK_PROB = "_block_prob"             # TODO

    # ========traffic param=====
    DENSITY = "_density"                   # ok
    MAX_SPEED = "_max_speed"               # TODO


class AlterParamEnv(MetaDriveEnv):
    
    def __init__(self, config: dict=None):
        self.env_params = {}
        for key in list(config.keys()):
            if key in EnvParams.__dict__.values():
                value = config.pop(key)
                self.env_params[key] = value
        super(AlterParamEnv, self).__init__(config)
    
    def setup_engine(self):
        super(MetaDriveEnv, self).setup_engine()
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("q", self.switch_to_third_person_view)
        from metadrive.manager.traffic_param_manager import TrafficParamManager
        from metadrive.manager.map_param_manager import MapParamManager
        self.engine.register_manager("map_manager", MapParamManager(self.env_params))
        self.engine.register_manager("traffic_manager", TrafficParamManager(self.env_params))

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
    config.update({EnvParams.LANE_WIDTH: 10, EnvParams.LANE_NUM: 10, EnvParams.BLOCK_NUM: 10, EnvParams.DENSITY: 0.9})
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation", type=str, default="lidar", choices=["lidar", "rgb_camera"])
    args = parser.parse_args()
    if args.observation == "rgb_camera":
        config.update(dict(offscreen_render=True))
    env = AlterParamEnv(config)
    try:
        o = env.reset()
        print(HELP_MESSAGE)
        env.vehicle.expert_takeover = True
        if args.observation == "rgb_camera":
            assert isinstance(o, dict)
            print("The observation is a dict with numpy arrays as values: ", {k: v.shape for k, v in o.items()})
        else:
            assert isinstance(o, np.ndarray)
            print("The observation is an numpy array with shape: ", o.shape)
        for i in range(1, 1000000000):
            o, r, d, info = env.step([0, 0])
            env.render(
                text={
                    "Auto-Drive (Switch mode: T)": "on" if env.current_track_vehicle.expert_takeover else "off",
                }
            )
            if d and info["arrive_dest"]:
                env.reset()
                env.current_track_vehicle.expert_takeover = True
    except:
        pass
    finally:
        env.close()

        