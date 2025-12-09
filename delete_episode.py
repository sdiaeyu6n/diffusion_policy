import os
import argparse
import shutil
from diffusion_policy.common.replay_buffer import ReplayBuffer

parser = argparse.ArgumentParser()
parser.add_argument("--zarr", required=True,
                    help="Path to replay_buffer.zarr (e.g. data/demo_pusht_real_lite6/replay_buffer.zarr)")
parser.add_argument("--episode", type=int, required=True,
                    help="Episode index to delete (0-based)")
args = parser.parse_args()

zarr_path = os.path.abspath(args.zarr)
tmp_path = zarr_path + ".tmp"

print("Source zarr:", zarr_path)

# open source buffer (read-only)
src = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="r")

# create new temp buffer
if os.path.exists(tmp_path):
    shutil.rmtree(tmp_path)
dst = ReplayBuffer.create_from_path(zarr_path=tmp_path, mode="a")

# copy all episodes except the one we want to drop
for i in range(src.n_episodes):
    if i == args.episode:
        print(f"Skipping episode {i}")
        continue
    ep = src.get_episode(i)
    dst.add_episode(ep, compressors="disk")
    print(f"Copied episode {i}")

# swap buffers
shutil.rmtree(zarr_path)
shutil.move(tmp_path, zarr_path)
print(f"Episode {args.episode} removed from {zarr_path}")
