# gello/robots/dex1_ros2_gripper.py

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from unitree_go.msg import MotorCmd, MotorCmds, MotorStates


class Dex1Ros2Gripper:
    """
    RobotiqGripper-compatible adapter for Unitree Dex1.

    GELLO/Robotiq convention:
        0   = fully open
        255 = fully closed

    Dex1 convention after calibration:
        q_closed is normally near 0 rad
        q increases as the gripper opens
    """

    def __init__(
        self,
        side: str = "left",
        q_closed: float = 0.0,
        q_open: float = 5.5,
        kp: float = 5.0,
        kd: float = 0.05,
        publish_hz: float = 100.0,
        command_watchdog_s: float = 0.2,
        state_timeout_s: float = 1.0,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")

        if q_open == q_closed:
            raise ValueError("q_open and q_closed must be different")

        if not rclpy.ok():
            rclpy.init(args=None)

        self.side = side
        self.q_closed = float(q_closed)
        self.q_open = float(q_open)
        self.kp = float(kp)
        self.kd = float(kd)
        self.command_watchdog_s = float(command_watchdog_s)
        self.state_timeout_s = float(state_timeout_s)

        self._lock = threading.Lock()
        self._state_event = threading.Event()

        self._current_q: Optional[float] = None
        self._last_state_time = 0.0

        self._target_q: Optional[float] = None
        self._last_command_time = 0.0

        node_name = f"dex1_{side}_gello_{os.getpid()}"
        self._node = Node(node_name)

        # A best-effort reader can receive from both best-effort and reliable
        # DDS state publishers.
        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Reliable publication is normally compatible with reliable or
        # best-effort DDS readers.
        command_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self._publisher = self._node.create_publisher(
            MotorCmds,
            f"/dex1/{side}/cmd",
            command_qos,
        )

        self._subscription = self._node.create_subscription(
            MotorStates,
            f"/dex1/{side}/state",
            self._state_callback,
            state_qos,
        )

        self._timer = self._node.create_timer(
            1.0 / publish_hz,
            self._publish_command,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

        self._spin_thread = threading.Thread(
            target=self._executor.spin,
            name=f"dex1-{side}-ros2",
            daemon=True,
        )
        self._spin_thread.start()

        if not self._state_event.wait(timeout=3.0):
            raise RuntimeError(
                f"No state received from /dex1/{side}/state. "
                "Check DDS interface, ROS_DOMAIN_ID and QoS."
            )

        print(
            f"Dex1 {side} connected: "
            f"q={self._current_q:.3f}, "
            f"range=[{self.q_closed:.3f}, {self.q_open:.3f}]"
        )

    def _state_callback(self, msg: MotorStates) -> None:
        if not msg.states:
            return

        with self._lock:
            self._current_q = float(msg.states[0].q)
            self._last_state_time = time.monotonic()

        self._state_event.set()

    def _publish_command(self) -> None:
        now = time.monotonic()

        with self._lock:
            target_q = self._target_q
            command_age = now - self._last_command_time

        # GELLO stops updating -> stop publishing -> Dex1 service enters brake.
        if target_q is None or command_age > self.command_watchdog_s:
            return

        command = MotorCmd()
        command.mode = 1
        command.q = float(target_q)
        command.dq = 0.0
        command.tau = 0.0
        command.kp = self.kp
        command.kd = self.kd
        command.reserve = [0, 0, 0]

        message = MotorCmds()
        message.cmds = [command]

        self._publisher.publish(message)

    def _normalized_to_q(self, position: float) -> float:
        """
        position:
            0.0 = open
            1.0 = closed
        """
        position = min(max(float(position), 0.0), 1.0)

        return self.q_open + position * (self.q_closed - self.q_open)

    def _q_to_normalized(self, q: float) -> float:
        """
        Dex1 q -> GELLO/Robotiq normalized position.

        Return:
            0.0 = open
            1.0 = closed
        """
        position = (self.q_open - q) / (self.q_open - self.q_closed)
        return min(max(position, 0.0), 1.0)

    def get_current_position(self) -> int:
        """
        Robotiq-compatible result:
            0   = open
            255 = closed
        """
        with self._lock:
            q = self._current_q
            state_age = time.monotonic() - self._last_state_time

        if q is None:
            raise RuntimeError("Dex1 state has not been received")

        if state_age > self.state_timeout_s:
            raise RuntimeError(
                f"Dex1 state timeout: last state was {state_age:.3f}s ago"
            )

        normalized = self._q_to_normalized(q)
        return int(round(normalized * 255.0))

    def move(
        self,
        position: int,
        speed: int = 255,
        force: int = 10,
    ) -> tuple[bool, int]:
        """
        Robotiq-compatible non-blocking move.

        speed and force are retained for interface compatibility. Dex1 uses
        q/dq/tau/kp/kd rather than Robotiq's speed/force command semantics.
        """
        del speed, force

        clipped_position = min(max(int(position), 0), 255)
        normalized = clipped_position / 255.0
        target_q = self._normalized_to_q(normalized)

        with self._lock:
            self._target_q = target_q
            self._last_command_time = time.monotonic()

        return True, clipped_position

    def close(self) -> None:
        self._executor.shutdown()
        self._node.destroy_node()
