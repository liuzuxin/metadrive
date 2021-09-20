import logging
from collections import namedtuple

import numpy as np

from metadrive.component.map.base_map import BaseMap
from metadrive.component.vehicle.vehicle_type import *
from metadrive.manager.base_manager import BaseManager
from metadrive.component.vehicle.vehicle_type import ReplayVehicle, SVehicle
from metadrive.policy.replay_policy import ReplayPolicy
from metadrive.policy.idm_policy import IDMPolicy

BlockVehicles = namedtuple("block_vehicles", "trigger_road vehicles")


class RealDataManager(BaseManager):
    VEHICLE_GAP = 10  # m

    def __init__(self):
        """
        Replay Argoverse data.
        """
        super(RealDataManager, self).__init__()
        self._traffic_vehicles = []

    def reset(self):
        """
        Generate traffic on map, according to the mode and density
        :return: List of Traffic vehicles
        """
        map = self.current_map
        self._create_argoverse_vehicles_once(map)

    def before_step(self):
        """
        All traffic vehicles make driving decision here
        :return: None
        """
        # trigger vehicles
        engine = self.engine

        for v in self._traffic_vehicles:
            p = self.engine.get_policy(v.name)
            v.before_step(p.act())
        return dict()

    def before_reset(self) -> None:
        """
        Clear the scene and then reset the scene to empty
        :return: None
        """
        super(RealDataManager, self).before_reset()
        self.density = self.engine.global_config["traffic_density"]
        self._traffic_vehicles = []

    def get_vehicle_num(self):
        """
        Get the vehicles on road
        :return:
        """
        return len(self._traffic_vehicles)

    def _create_argoverse_vehicles_once(self, map: BaseMap) -> None:
        """
        Trigger mode, vehicles will be triggered only once, and disappear when arriving destination
        :param map: Map map.road_network[index]
        :param traffic_density: it can be adjusted each episode
        :return: None
        """
        real_data_config = self.engine.global_config["real_data_config"]
        locate_info = real_data_config["locate_info"]
        source = real_data_config["source"]
        if source == 'tracking':
            self._create_from_tracking(locate_info, map)
        else:
            self._create_from_forecasting(locate_info)

    def _create_from_forecasting(self, locate_info):
        for key in locate_info.keys():
            this_info = locate_info[key]
            generated_v = self.spawn_object(
                SVehicle,
                vehicle_config={
                    "spawn_lane_index": this_info["spawn_lane_index"],
                    # "spawn_longitude": this_info["long"],
                    # "spawn_lateral": this_info["lat"],
                    "destination_node": this_info["targ_node"],
                }
            )
            generated_v.set_static(True)
            self.engine.add_policy(generated_v.id, IDMPolicy(generated_v, self.generate_seed()))
            self._traffic_vehicles.append(generated_v)

    def _filter_vehicle_configs(self, locate_info, max_to_keep=10):
        locate_info = dict(
            sorted(
                locate_info.items(),
                key=lambda x: np.linalg.norm(x[1]["init_pos"] - list(x[1]["traj"].values())[-1]),
                reverse=True
            )
        )
        key_to_pop = []
        for i, key in enumerate(locate_info.keys()):
            if i >= max_to_keep:
                key_to_pop.append(key)
        for config in self.potential_vehicle_configs:
            if config["id"] in key_to_pop:
                config["id"] = None

    def _create_from_tracking(self, locate_info, map):
        pos_dict = {i: j["init_pos"] for i, j in zip(locate_info.keys(), locate_info.values())}

        block = map.blocks[0]
        lanes = block.argo_lanes.values()
        roads = block.block_network.get_roads(direction='positive', lane_num=1)
        self.potential_vehicle_configs = []
        for l in lanes:
            start = np.max(l.centerline, axis=0)
            end = np.min(l.centerline, axis=0)
            for idx, pos in zip(pos_dict.keys(), pos_dict.values()):
                # if start[0] > pos[0] > end[0] and start[1] > pos[1] > end[1]:
                # if l.index is not None:
                v_type = self.random_vehicle_type(prob=[0.4, 0.3, 0.3, 0, 0])
                long, lat = l.local_coordinates(pos)
                config = {
                    "id": idx,
                    "type": v_type,
                    "v_config": {
                        "spawn_lane_index": (l.start_node, l.end_node, 0),
                        "spawn_longitude": 0,
                        "enable_reverse": False,
                    }
                }
                self.potential_vehicle_configs.append(config)
                pos_dict.pop(idx, None)
                break
        self._filter_vehicle_configs(locate_info, max_to_keep=10)
        for config in self.potential_vehicle_configs:
            if config["id"] is not None:
                v_config = config["v_config"]
                v_config.update(self.engine.global_config["traffic_vehicle_config"])
                generated_v = self.spawn_object(config["type"], vehicle_config=v_config)
                generated_v.set_static(True)
                generated_v.set_position(generated_v.position)
                self.engine.add_policy(generated_v.id, ReplayPolicy(generated_v, locate_info[config["id"]]))
                self._traffic_vehicles.append(generated_v)

    def random_vehicle_type(self, prob=[0.2, 0.3, 0.3, 0.2, 0]):
        # vehicle_type = random_vehicle_type(self.np_random, prob)
        vehicle_type = ReplayVehicle
        return vehicle_type

    def destroy(self) -> None:
        """
        Destory func, release resource
        :return: None
        """
        self.clear_objects([v.id for v in self._traffic_vehicles])
        self._traffic_vehicles = []
        # current map

        # traffic vehicle list
        self._traffic_vehicles = None

    def __del__(self):
        logging.debug("{} is destroyed".format(self.__class__.__name__))

    def __repr__(self):
        return self.vehicles.__repr__()

    @property
    def vehicles(self):
        return list(self.engine.get_objects(filter=lambda o: isinstance(o, BaseVehicle)).values())

    @property
    def traffic_vehicles(self):
        return list(self._traffic_vehicles)

    @property
    def current_map(self):
        return self.engine.map_manager.current_map
