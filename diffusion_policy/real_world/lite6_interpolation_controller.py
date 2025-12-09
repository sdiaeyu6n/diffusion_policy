import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from xarm.wrapper import XArmAPI

from diffusion_policy.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty
)
from diffusion_policy.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer
)
from diffusion_policy.common.pose_trajectory_interpolator import (
    PoseTrajectoryInterpolator
)


class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


# === UF Studio "initial/zero" pose joint configuration (rad) ===
INITIAL_JOINTS = np.array(
    [
        5.6931306e-02,   # J1 ≈ 0.057 rad
        1.5230398e-03,   # J2
        1.5256709e-03,   # J3
        3.5003118e-02,   # J4
        -8.1976541e-06,  # J5
        2.2030687e-02,   # J6
    ],
    dtype=np.float64,
)


class Lite6InterpolationController(mp.Process):
    """
    UR5용 RTDEInterpolationController와 동일한 public 인터페이스를 가지는
    Lite6(xArm) 전용 컨트롤러.

    - RealEnv에서 그대로 사용 가능하도록, __init__ 시그니처를 맞춤.
    - 내부 표현은 pose [x,y,z, r1,r2,r3] with position in meters.
    - xArm에는 vc_set_cartesian_velocity(mm/s)로 명령 전송.
    """

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_ip,
        frequency: float = 125,
        lookahead_time: float = 0.1,   # UR에서만 쓰이지만 시그니처 유지
        gain: float = 300,             # UR에서만 쓰이지만 시그니처 유지
        max_pos_speed: float = 0.25,
        max_rot_speed: float = 0.16,
        launch_timeout: float = 3,
        tcp_offset_pose=None,          # 현재는 직접 적용하지 않고, xArm 쪽 TCP 세팅에 의존
        payload_mass=None,             # unused (to match UR API)
        payload_cog=None,              # unused
        joints_init=None,
        joints_init_speed: float = 1.05,   # unused (we use fixed speed)
        soft_real_time: bool = False,
        verbose: bool = False,
        receive_keys=None,
        get_max_k: int = 128,
    ):
        assert 0 < frequency <= 500
        assert max_pos_speed > 0
        assert max_rot_speed > 0

        if joints_init is None:
            joints_init = INITIAL_JOINTS.copy()
        else:
            joints_init = np.array(joints_init, dtype=np.float64)
        assert joints_init.shape == (6,)

        super().__init__(name="Lite6InterpolationController")
        self.robot_ip = robot_ip
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        self.gain = gain
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.launch_timeout = launch_timeout
        self.tcp_offset_pose = tcp_offset_pose
        self.payload_mass = payload_mass
        self.payload_cog = payload_cog
        self.joints_init = joints_init
        self.joints_init_speed = joints_init_speed
        self.soft_real_time = soft_real_time
        self.verbose = verbose

        # ===== 입력 큐 (UR 버전과 동일한 구조) =====
        example_msg = {
            "cmd": Command.SERVOL.value,
            "target_pose": np.zeros((6,), dtype=np.float64),  # [x,y,z, r1,r2,r3]
            "duration": 0.0,
            "target_time": 0.0,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example_msg,
            buffer_size=256,
        )

        # ===== 상태 링버퍼 (RealEnv와 호환되도록 키 맞춤) =====
        if receive_keys is None:
            receive_keys = [
                "ActualTCPPose",
                "ActualTCPSpeed",
                "ActualQ",
                "ActualQd",
                "TargetTCPPose",
                "TargetTCPSpeed",
                "TargetQ",
                "TargetQd",
            ]

        examples_state = {}
        for key in receive_keys:
            examples_state[key] = np.zeros((6,), dtype=np.float64)
        # timestamp
        examples_state["robot_receive_timestamp"] = np.array(0.0, dtype=np.float64)

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=examples_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency,
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys

    # ========= launch / stop ===========
    def start(self, wait: bool = True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[Lite6InterpolationController] Controller process spawned at {self.pid}")

    def stop(self, wait: bool = True):
        message = {"cmd": Command.STOP.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive(), "[Lite6InterpolationController] Failed to start controller."

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration: float = 0.1):
        """
        pose: 6D [x,y,z, r1,r2,r3] (meters, radians)
        duration: 이 pose까지 도달하라고 원하는 시간 (sec)
        """
        assert self.is_alive()
        assert duration >= (1.0 / self.frequency)
        pose = np.array(pose, dtype=np.float64)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SERVOL.value,
            "target_pose": pose,
            "duration": float(duration),
        }
        self.input_queue.put(message)

    def schedule_waypoint(self, pose, target_time: float):
        """
        target_time: wall-clock time (time.time()) 기준의 미래 시각
        """
        assert target_time > time.time()
        pose = np.array(pose, dtype=np.float64)
        assert pose.shape == (6,)

        message = {
            "cmd": Command.SCHEDULE_WAYPOINT.value,
            "target_pose": pose,
            "target_time": float(target_time),
        }
        self.input_queue.put(message)

    # ========= read APIs ============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main process loop ============
    def run(self):
        # soft real-time (옵션)
        if self.soft_real_time:
            try:
                os.sched_setscheduler(
                    0, os.SCHED_RR, os.sched_param(20)
                )
            except PermissionError:
                if self.verbose:
                    print("[Lite6InterpolationController] Failed to set real-time scheduler (permission).")

        # ===== xArm 연결 =====
        arm = XArmAPI(self.robot_ip, is_radian=True)

        try:
            if self.verbose:
                print(f"[Lite6InterpolationController] Connecting to robot at {self.robot_ip}")

            # 기본 설정
            arm.motion_enable(True)
            arm.set_mode(0)
            arm.set_state(0)
            time.sleep(1.0)

            # 초기 관절 자세로 이동 (UF Studio initial pose)
            arm.set_servo_angle_j(
                self.joints_init.tolist(),
                speed=0.7,
                accel=1.0,
                is_radian=True,
                wait=True,
            )

            # velocity mode
            arm.set_mode(5)
            arm.set_state(0)
            time.sleep(1.0)

            # ===== Trajectory 초기화 =====
            dt = 1.0 / self.frequency

            # 현재 TCP pose (x,y,z mm + orientation rad) 읽기
            ret, pose_mm = arm.get_position()
            pose_mm = np.array(pose_mm, dtype=np.float64)

            # 내부 표현: position in meters, orientation 그대로
            curr_pose = pose_mm.copy()
            curr_pose[:3] /= 1000.0  # mm → m

            curr_t = time.monotonic()
            last_waypoint_time = curr_t

            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose],
            )

            iter_idx = 0
            keep_running = True

            while keep_running:
                loop_start = time.perf_counter()
                t_now = time.monotonic()

                # 현재 시각 기준 목표 pose (m 단위)
                target_pose = pose_interp(t_now)

                # 실제 pose 읽기
                ret, pose_mm = arm.get_position()
                pose_mm = np.array(pose_mm, dtype=np.float64)
                actual_pose = pose_mm.copy()
                actual_pose[:3] /= 1000.0  # mm → m

                # 위치 차이
                dpos = target_pose[:3] - actual_pose[:3]

                # 회전은 일단 고정(orientation 유지)
                drot = np.zeros(3, dtype=np.float64)

                speed = np.linalg.norm(dpos) / dt
                if speed > 1e-6:
                    scale = min(self.max_pos_speed / speed, 1.0)
                    vel_m_per_s = dpos * scale / dt
                else:
                    vel_m_per_s = np.zeros_like(dpos)

                # m/s → mm/s
                cart_vel_mm = vel_m_per_s * 1000.0
                vx, vy, vz = cart_vel_mm
                rx_vel, ry_vel, rz_vel = 0.0, 0.0, 0.0

                try:
                    arm.vc_set_cartesian_velocity(
                        [vx, vy, vz, rx_vel, ry_vel, rz_vel]
                    )
                except Exception as e:
                    if self.verbose:
                        print("[Lite6InterpolationController] vc_set_cartesian_velocity error:", e)

                # ===== 상태 로깅 =====
                state = {}

                for key in self.receive_keys:
                    if key == "ActualTCPPose":
                        state[key] = actual_pose.copy()
                    elif key == "TargetTCPPose":
                        state[key] = target_pose.copy()
                    elif key in ("ActualTCPSpeed", "TargetTCPSpeed"):
                        v_m = vel_m_per_s
                        state[key] = np.concatenate([v_m, drot])
                    elif key in ("ActualQ", "TargetQ"):
                        _, q = arm.get_servo_angle(is_radian=True)
                        q = np.array(q[:6], dtype=np.float64)
                        state[key] = q
                    elif key in ("ActualQd", "TargetQd"):
                        # joint velocity가 필요하면 여기서 채움
                        state[key] = np.zeros((6,), dtype=np.float64)
                    else:
                        state[key] = np.zeros((6,), dtype=np.float64)

                state["robot_receive_timestamp"] = time.time()
                self.ring_buffer.put(state)

                # ===== 큐에서 명령 처리 =====
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands["cmd"])
                except Empty:
                    n_cmd = 0

                for i in range(n_cmd):
                    command = {k: v[i] for k, v in commands.items()}
                    cmd = command["cmd"]

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break

                    elif cmd == Command.SERVOL.value:
                        target_pose_cmd = command["target_pose"]
                        duration = float(command["duration"])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose_cmd,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print(
                                "[Lite6InterpolationController] New pose target:",
                                target_pose_cmd,
                                "duration:",
                                duration,
                            )

                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose_cmd = command["target_pose"]
                        target_time = float(command["target_time"])
                        # wall-clock → monotonic
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose_cmd,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time,
                        )
                        last_waypoint_time = target_time

                    else:
                        keep_running = False
                        break

                # 주기 맞추기
                elapsed = time.perf_counter() - loop_start
                sleep_time = (1.0 / self.frequency) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    freq_actual = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                    print(
                        f"[Lite6InterpolationController] Actual frequency ~{freq_actual:.1f} Hz"
                    )

        finally:
            # mandatory cleanup
            try:
                arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])
            except Exception:
                pass

            try:
                arm.disconnect()
            except Exception:
                pass

            self.ready_event.set()
            if self.verbose:
                print(f"[Lite6InterpolationController] Disconnected from robot {self.robot_ip}")
