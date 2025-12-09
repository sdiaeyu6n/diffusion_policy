"""
Lite6 + RealSense recorder for Diffusion Policy style dataset.

- Control: SpaceMouse -> Lite6 (Cartesian velocity, like demo_xarm.py)
- Sensing: MultiRealsense (top camera, wrist camera, etc.)
- Storage:
  - replay_buffer.zarr (episode-wise state/action/stage/timestamp)
  - videos/<episode_id>/<camera_idx>.mp4 (H.264)

Controls (keyboard on OpenCV window):
- 'C': start recording an episode
- 'S': stop current recording
- 'Q': quit program
- Backspace: delete the most recently recorded episode

Controls (SpaceMouse):
- Translation on XYZ
- (You can easily extend to rotation later if needed)
"""

import os
import time
import shutil
from multiprocessing.managers import SharedMemoryManager

import click
import cv2
import numpy as np
import scipy.spatial.transform as st

from xarm.wrapper import XArmAPI

from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter,
    Key,
    KeyCode,
)
from diffusion_policy.common.precise_sleep import precise_wait

from diffusion_policy.real_world.multi_realsense import (
    MultiRealsense,
    SingleRealsense,
)
from diffusion_policy.real_world.video_recorder import VideoRecorder
from diffusion_policy.real_world.multi_camera_visualizer import MultiCameraVisualizer
from diffusion_policy.common.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols


# ---- Custom initial/reset pose (same as your Studio screenshot) ----
# [x(mm), y(mm), z(mm), roll(rad), pitch(rad), yaw(rad)]
HOME_POSE = np.array([
    124.1,      # X
    0.0,        # Y
    173.9,      # Z  (raise this value if you want it higher)
    np.pi,      # Roll 180°
    0.0,        # Pitch
    0.0         # Yaw
], dtype=float)

MIN_Z = HOME_POSE[2]      # never go below this


@click.command()
@click.option(
    "--output",
    "-o",
    required=True,
    help="Directory to save demonstration dataset (will contain replay_buffer.zarr and videos/).",
)
@click.option(
    "--robot_ip",
    "-ri",
    required=True,
    help="Lite6 IP address.",
)
@click.option(
    "--frequency",
    "-f",
    default=20.0,
    type=float,
    help="Control frequency in Hz.",
)
@click.option(
    "--command_latency",
    "-cl",
    default=0.02,
    type=float,
    help="Latency between reading SpaceMouse and applying command (sec).",
)
@click.option(
    "--max_speed",
    "-ms",
    default=0.10,
    type=float,
    help="Approx linear speed scale in m/s.",
)
@click.option(
    "--video_fps",
    default=30,
    type=int,
    help="RealSense capture FPS.",
)
@click.option(
    "--video_resolution",
    default="1280x720",
    help="RealSense color resolution as WIDTHxHEIGHT (e.g. 1280x720).",
)
def main(
    output,
    robot_ip,
    frequency,
    command_latency,
    max_speed,
    video_fps,
    video_resolution,
):
    dt = 1.0 / frequency
    out_dir = os.path.abspath(output)
    os.makedirs(out_dir, exist_ok=True)

    # Parse resolution string
    try:
        w_str, h_str = video_resolution.lower().split("x")
        video_res = (int(w_str), int(h_str))
    except Exception:
        raise ValueError(
            f"Invalid --video_resolution '{video_resolution}'. Use e.g. 1280x720."
        )

    # Prepare replay buffer
    zarr_path = os.path.join(out_dir, "replay_buffer.zarr")
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=zarr_path, mode="a")

    # Directory for videos
    video_root = os.path.join(out_dir, "videos")
    os.makedirs(video_root, exist_ok=True)

    with SharedMemoryManager() as shm_manager:
        # --- Set up SpaceMouse + keyboard ---
        with KeystrokeCounter() as key_counter, Spacemouse(
            shm_manager=shm_manager
        ) as sm:
            cv2.setNumThreads(1)

            # --- Connect Lite6 ---
            arm = XArmAPI(robot_ip, is_radian=True)
            arm.motion_enable(True)
            time.sleep(1.0)

            # Go directly to our custom initial pose (NO arm.reset())
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(0.5)
            arm.set_position(
                x=float(HOME_POSE[0]),
                y=float(HOME_POSE[1]),
                z=float(HOME_POSE[2]),
                roll=float(HOME_POSE[3]),
                pitch=float(HOME_POSE[4]),
                yaw=float(HOME_POSE[5]),
                is_radian=True,
                wait=True,
            )

            arm.set_mode(5)  # Cartesian velocity control
            arm.set_state(0)
            time.sleep(1.0)

            # --- Set up RealSense (MultiRealsense) ---
            camera_serials = SingleRealsense.get_connected_devices_serial()
            if len(camera_serials) == 0:
                raise RuntimeError("No RealSense cameras found.")

            print(f"[INFO] Found RealSense cameras: {camera_serials}")

            # Video recorder (H.264)
            video_recorder = VideoRecorder.create_h264(
                fps=video_fps,
                codec="h264",
                input_pix_fmt="bgr24",
                crf=21,
                thread_type="FRAME",
                thread_count=3,
            )

            # Transform for resizing color image for visualization
            # (use original resolution for recording)
            obs_image_resolution = video_res
            color_tf = get_image_transform(
                input_res=video_res,
                output_res=obs_image_resolution,
                bgr_to_rgb=False,  # for visualization (OpenCV expects BGR)
            )

            # Multi-camera layout for visualizer
            rw, rh, col, row = optimal_row_cols(
                n_cameras=len(camera_serials),
                in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
                max_resolution=(1280, 720),
            )

            vis_color_tf = get_image_transform(
                input_res=video_res,
                output_res=(rw, rh),
                bgr_to_rgb=False,
            )

            def transform(data):
                data["color"] = data["color"]  # keep raw for recording
                return data

            def vis_transform(data):
                data["color"] = vis_color_tf(data["color"])
                return data

            realsense = MultiRealsense(
                serial_numbers=camera_serials,
                shm_manager=shm_manager,
                resolution=video_res,
                capture_fps=video_fps,
                put_fps=video_fps,
                put_downsample=False,
                record_fps=video_fps,
                enable_color=True,
                enable_depth=False,
                enable_infrared=False,
                get_max_k=100,
                transform=transform,
                vis_transform=vis_transform,
                recording_transform=None,
                video_recorder=video_recorder,
                verbose=False,
            )

            multi_vis = MultiCameraVisualizer(
                realsense=realsense,
                row=row,
                col=col,
                rgb_to_bgr=False,
            )

            # Start camera + visualizer
            realsense.start(wait=True)
            multi_vis.start(wait=True)

            # Wait a bit
            time.sleep(1.0)
            print("[INFO] System ready. Press 'C' in the OpenCV window to start recording.")

            # State for recording
            obs_acc = None
            act_acc = None
            stage_acc = None
            current_episode_id = replay_buffer.n_episodes

            last_realsense_data = None

            # Helper to start an episode
            def start_episode():
                nonlocal obs_acc, act_acc, stage_acc, current_episode_id

                episode_id = replay_buffer.n_episodes
                current_episode_id = episode_id
                episode_video_dir = os.path.join(video_root, str(episode_id))
                os.makedirs(episode_video_dir, exist_ok=True)

                # Build video paths (per camera)
                video_paths = []
                for cam_idx in range(len(camera_serials)):
                    video_paths.append(
                        os.path.join(episode_video_dir, f"{cam_idx}.mp4")
                    )

                start_time = time.time()
                # (Re)start realsense put & recording
                realsense.restart_put(start_time=start_time)
                realsense.start_recording(
                    video_path=video_paths,
                    start_time=start_time,
                )

                obs_acc = TimestampObsAccumulator(
                    start_time=start_time,
                    dt=1.0 / frequency,
                )
                act_acc = TimestampActionAccumulator(
                    start_time=start_time,
                    dt=1.0 / frequency,
                )
                stage_acc = TimestampActionAccumulator(
                    start_time=start_time,
                    dt=1.0 / frequency,
                )

                print(f"[INFO] Episode {episode_id} started.")

            # Helper to finish an episode
            def finish_episode():
                nonlocal obs_acc, act_acc, stage_acc, current_episode_id

                realsense.stop_recording()

                if obs_acc is None:
                    return

                obs_data = obs_acc.data
                obs_ts = obs_acc.timestamps

                actions = act_acc.actions
                act_ts = act_acc.timestamps

                stages = stage_acc.actions

                n_steps = min(len(obs_ts), len(act_ts))
                if n_steps > 0:
                    episode = dict()
                    episode["timestamp"] = obs_ts[:n_steps]
                    episode["action"] = actions[:n_steps]
                    episode["stage"] = stages[:n_steps]

                    for k, v in obs_data.items():
                        episode[k] = v[:n_steps]

                    replay_buffer.add_episode(episode, compressors="disk")
                    ep_id = replay_buffer.n_episodes - 1
                    print(f"[INFO] Episode {ep_id} saved.")

                obs_acc = None
                act_acc = None
                stage_acc = None

            # Helper to delete last episode
            def drop_last_episode():
                nonlocal current_episode_id

                finish_episode()
                replay_buffer.drop_episode()
                ep_id = replay_buffer.n_episodes
                ep_video_dir = os.path.join(video_root, str(ep_id))
                if os.path.exists(ep_video_dir):
                    shutil.rmtree(ep_video_dir)
                print(f"[INFO] Episode {ep_id} dropped.")
                current_episode_id = replay_buffer.n_episodes

            # Main control loop
            err, pose = arm.get_position(is_radian=True)
            # pose: [x(mm), y(mm), z(mm), roll(rad), pitch(rad), yaw(rad)]
            target_pose = np.array(pose, dtype=float)

            t_start = time.monotonic()
            idx = 0
            running = True
            recording = False

            while running:
                t_cycle_end = t_start + (idx + 1) * dt
                t_sample = t_cycle_end - command_latency

                # handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char="q"):
                        if recording:
                            finish_episode()
                            recording = False
                        running = False
                    elif key_stroke == KeyCode(char="c"):
                        if not recording:
                            start_episode()
                            recording = True
                    elif key_stroke == KeyCode(char="s"):
                        if recording:
                            finish_episode()
                            recording = False
                    elif key_stroke == Key.backspace:
                        if click.confirm(
                            "Are you sure you want to drop the last episode?"
                        ):
                            drop_last_episode()
                            recording = False

                stage = key_counter[Key.space]

                precise_wait(t_sample)

                # Get latest frames from all cameras
                last_realsense_data = realsense.get(
                    k=1, out=last_realsense_data
                )

                # Visualization: show camera_0 as Top, camera_1 as Wrist (if exists)
                if 0 in last_realsense_data:
                    img0 = last_realsense_data[0]["color"][-1]
                    top_img = color_tf(img0)
                    text = f"Episode: {current_episode_id}, Stage: {stage}"
                    if recording:
                        text += ", Recording!"
                    cv2.putText(
                        top_img,
                        text,
                        (10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        thickness=2,
                        color=(255, 255, 255),
                    )
                    cv2.imshow("Top", top_img)

                if 1 in last_realsense_data:
                    img1 = last_realsense_data[1]["color"][-1]
                    wrist_img = color_tf(img1)
                    cv2.imshow("Wrist", wrist_img)

                cv2.pollKey()

                # SpaceMouse control (translation only for now)
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]  # xyz
                drot_xyz = sm_state[3:] * 0.0  # ignore rotation

                lin_vel = dpos * max_speed  # m/s
                ang_vel = np.zeros(3)

                # Reset pose when left button is pressed (NO arm.reset())
                if sm.is_button_pressed(0):
                    arm.set_mode(0)
                    arm.set_state(0)
                    time.sleep(0.5)
                    arm.set_position(
                        x=float(HOME_POSE[0]),
                        y=float(HOME_POSE[1]),
                        z=float(HOME_POSE[2]),
                        roll=float(HOME_POSE[3]),
                        pitch=float(HOME_POSE[4]),
                        yaw=float(HOME_POSE[5]),
                        is_radian=True,
                        wait=True,
                    )
                    arm.set_mode(5)
                    arm.set_state(0)
                    time.sleep(0.5)
                    err, pose = arm.get_position(is_radian=True)
                    target_pose = np.array(pose, dtype=float)
                    lin_vel[:] = 0.0
                    ang_vel[:] = 0.0

                vx, vy, vz = lin_vel
                wx, wy, wz = ang_vel

                # --- Z safety: do not go below MIN_Z ---
                _, pose_now = arm.get_position(is_radian=True)
                z_now = pose_now[2]
                if z_now <= MIN_Z and vz < 0:
                    vz = 0.0

                # Lite6 expects mm/s in vc_set_cartesian_velocity
                scale = 1000.0
                v_cmd = [vx * scale, vy * scale, vz * scale, wx, wy, wz]
                arm.vc_set_cartesian_velocity(v_cmd)

                # Build robot_eef_pose in UR style [x,y,z,rx,ry,rz]
                # get current pose
                _, pose_now = arm.get_position(is_radian=True)
                x_mm, y_mm, z_mm, roll, pitch, yaw = pose_now
                pos_m = np.array([x_mm, y_mm, z_mm], dtype=np.float32) / 1000.0
                rot = st.Rotation.from_euler("xyz", [roll, pitch, yaw])
                rotvec = rot.as_rotvec().astype(np.float32)
                robot_eef_pose = np.concatenate([pos_m, rotvec])

                now_ts = time.time()

                # Record into accumulators (if recording)
                if recording and obs_acc is not None:
                    obs_acc.put(
                        {"robot_eef_pose": robot_eef_pose[None, :]},
                        np.array([now_ts]),
                    )
                    # For action, we simply store eef pose as well
                    act_acc.put(
                        robot_eef_pose[None, :],
                        np.array([now_ts]),
                    )
                    stage_acc.put(
                        np.array([stage], dtype=np.int64),
                        np.array([now_ts]),
                    )

                precise_wait(t_cycle_end)
                idx += 1

            # Cleanup
            print("[INFO] Stopping robot and RealSense...")
            arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
            arm.disconnect()
            realsense.stop(wait=True)
            multi_vis.stop(wait=True)
            cv2.destroyAllWindows()
            print("[INFO] Done.")


if __name__ == "__main__":
    main()
