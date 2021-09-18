"""
This script demonstrates how to use the environment where traffic and road map are loaded from argoverse dataset.
"""
from metadrive.envs.argoverse_env import ArgoverseEnv, ArgoverseGeneralizationEnv
from panda3d.core import PNMImage
import logging
import os
import argparse
# logging.basicConfig(level=logging.DEBUG)
parser = argparse.ArgumentParser()
parser.add_argument("--log_id", type=str, default="6db21fda-80cd-3f85-b4a7-0aadeb14724d")
args = parser.parse_args()

if __name__ == "__main__":
    print("We are preparing argoverse environment!")
    # env = ArgoverseEnv(None, {"manual_control": True, "use_render": True})
    log_id = args.log_id.split(".")[0]
    # env = ArgoverseGeneralizationEnv(log_id, {"manual_control": True, "use_render": True})
    env = ArgoverseEnv(log_id, {"manual_control": True, "use_render": True})
    o = env.reset()
    for i in range(1, 300):
        # if not os.path.exists("video/{}".format(log_id)):
        #     os.mkdir("video/{}".format(log_id))
        # img = PNMImage()
        # env.engine.win.getScreenshot(img)
        # img.write("video/{}/{}.png".format(log_id, i))
        # if not os.path.exists("top_down_view"):
        #     os.mkdir("top_down_view")
        # if i == 8:
        #     env.switch_to_top_down_view()
        # if i == 10:
        #     img = PNMImage()
        #     env.engine.win.getScreenshot(img)
        #     img.write("top_down_view/{}.png".format(log_id))
        #     break
        o, r, d, info = env.step([0., 0.])
        print(r, d)
        print(i)
    env.close()
