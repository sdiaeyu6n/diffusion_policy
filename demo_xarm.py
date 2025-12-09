# demo_xarm.py
import time
import sys
import numpy as np
import scipy.spatial.transform as st
import click
import cv2

from multiprocessing.managers import SharedMemoryManager

from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.real_world.keystroke_counter import (
    KeystrokeCounter, Key
)
from diffusion_policy.common.precise_sleep import precise_wait

# xArm SDK
from xarm.wrapper import XArmAPI

# ---------------------------------------------------------------------------
# Custom "home" pose = the pose from your screenshot
# [x, y, z, roll, pitch, yaw]
# xyz: mm, rpy: radians (because we use XArmAPI(is_radian=True))
# ---------------------------------------------------------------------------
HOME_POSE = np.array([
    124.1,      # X (mm)
    0.0,        # Y (mm)
    173.9,      # Z (mm)  <-- raise this if you want it higher
    np.pi,      # Roll 180° in radians
    0.0,        # Pitch
    0.0         # Yaw
], dtype=float)

# Do not allow the tool to go below this Z
MIN_Z = HOME_POSE[2]


def move_to_home(arm: XArmAPI):
    """Move robot to our custom home pose in Cartesian space."""
    # switch to position mode
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
        is_radian=True,   # because we created XArmAPI(is_radian=True)
        wait=True
    )

    # back to Cartesian velocity mode
    arm.set_mode(5)
    arm.set_state(0)
    time.sleep(0.5)


@click.command()
@click.option('--robot_ip', '-ri', required=True, help="Lite6 IP address")
@click.option('--frequency', '-f', default=50.0, type=float)
@click.option('--command_latency', '-cl', default=0.02, type=float)
@click.option('--max_speed', '-ms', default=0.1, type=float,  # m/s-ish
              help="Approx Cartesian speed scale")
def main(robot_ip, frequency, command_latency, max_speed):
    dt = 1.0 / frequency

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
             Spacemouse(shm_manager=shm_manager) as sm:

            cv2.setNumThreads(1)

            # --- connect Lite6 ---
            arm = XArmAPI(robot_ip, is_radian=True)
            arm.motion_enable(True)
            time.sleep(1.0)

            # Go to our custom initial pose (same as Studio screenshot)
            move_to_home(arm)

            print("Ready. Use SpaceMouse to move, Ctrl+C to quit.")
            _, pose = arm.get_position()   # [x, y, z, roll, pitch, yaw] (mm, rad)
            target_pose = np.array(pose, dtype=float)

            t_start = time.monotonic()
            idx = 0
            running = True

            while running:
                t_cycle_end = t_start + (idx + 1) * dt
                t_sample = t_cycle_end - command_latency

                # handle key presses
                press_events = key_counter.get_press_events()
                if Key.esc in press_events:
                    running = False

                precise_wait(t_sample)

                # SpaceMouse delta in 6D
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3]      # translation
                drot_xyz = sm_state[3:]  # rotation; ignore for now

                # scale to velocity
                lin_vel = dpos * max_speed     # ~ m/s scale
                ang_vel = np.zeros(3)          # ignore rotation for now

                # reset pose when SpaceMouse left button pressed
                if sm.is_button_pressed(0):
                    move_to_home(arm)
                    _, pose = arm.get_position()
                    target_pose = np.array(pose, dtype=float)
                    lin_vel[:] = 0
                    ang_vel[:] = 0

                vx, vy, vz = lin_vel
                wx, wy, wz = ang_vel

                # --- Z safety: don't go below MIN_Z ---
                _, pose_now = arm.get_position()
                z_now = pose_now[2]
                if z_now <= MIN_Z and vz < 0:
                    vz = 0.0

                # Lite6 expects mm/s in Cartesian velocity interface,
                # so scale if your units are meters:
                scale = 1000.0
                v_cmd = [vx * scale, vy * scale, vz * scale,
                         wx, wy, wz]

                arm.vc_set_cartesian_velocity(v_cmd)

                precise_wait(t_cycle_end)
                idx += 1

            # stop
            arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
            arm.disconnect()


if __name__ == "__main__":
    main()
