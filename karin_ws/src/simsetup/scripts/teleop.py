#!/usr/bin/env python3
import math
import sys
import threading
import asyncio
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from pynput import keyboard
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed

class WAMVTeleop(Node):
    def __init__(self):
        super().__init__('wamv_teleop')

        # Control parameters - TUNED FOR STABILITY
        self.MAX_VEL = 1.0           # m/s max velocity command
        self.MAX_YAWRATE = 30.0      # deg/s max yaw rate
        self.VEL_TO_THRUST = 200.0   # N per m/s (reduced)
        self.YAW_TO_THRUST = 3.0     # N per deg/s
        self.MAX_THRUST = 250.0      # Absolute thrust limit
        self.DT = 0.05

        # State
        self.vel_cmd = 0.0
        self.yaw_cmd = 0.0
        self.offboard_active = False
        self.running = True
        self.keys = set()

        # Publishers
        self.left_pub = self.create_publisher(Float64, '/wamv/thrust/left', 10)
        self.right_pub = self.create_publisher(Float64, '/wamv/thrust/right', 10)

        # Keyboard
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

        # MAVSDK
        self.drone = System()
        self._mavsdk_loop = None

        # Control timer
        self.timer = self.create_timer(self.DT, self.control_loop)

        # Start offboard
        threading.Thread(target=self.mavsdk_thread, daemon=True).start()

        self.get_logger().info("=" * 50)
        self.get_logger().info("WAMV Offboard Teleop")
        self.get_logger().info("W/S: Forward/Back | A/D: Turn")
        self.get_logger().info("SPACE: Stop | Q: Quit")
        self.get_logger().info("=" * 50)

    def on_press(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.keys.add(key.char.lower())
        except:
            if key == keyboard.Key.space:
                self.keys.add('space')

    def on_release(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                self.keys.discard(key.char.lower())
        except:
            if key == keyboard.Key.space:
                self.keys.discard('space')

    def control_loop(self):
        if 'q' in self.keys:
            self.get_logger().info("Shutting down...")
            self.listener.stop()
            rclpy.shutdown()
            sys.exit(0)

        # Get velocity commands from keys
        if 'w' in self.keys:
            self.vel_cmd = self.MAX_VEL
        elif 's' in self.keys:
            self.vel_cmd = -self.MAX_VEL * 0.5
        else:
            self.vel_cmd = 0.0

        if 'a' in self.keys:
            self.yaw_cmd = self.MAX_YAWRATE
        elif 'd' in self.keys:
            self.yaw_cmd = -self.MAX_YAWRATE
        else:
            self.yaw_cmd = 0.0

        if 'space' in self.keys:
            self.vel_cmd = 0.0
            self.yaw_cmd = 0.0
            self.keys.discard('space')

        # Send to offboard (for QGC monitoring)
        if self.offboard_active and self._mavsdk_loop:
            asyncio.run_coroutine_threadsafe(
                self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(self.vel_cmd, 0.0, 0.0, self.yaw_cmd)
                ),
                self._mavsdk_loop
            )

        # Convert velocity to thrust (SIMPLE OPEN LOOP)
        base_thrust = self.vel_cmd * self.VEL_TO_THRUST
        yaw_thrust = self.yaw_cmd * self.YAW_TO_THRUST

        left = base_thrust - yaw_thrust
        right = base_thrust + yaw_thrust

        # Clamp
        left = max(-self.MAX_THRUST, min(self.MAX_THRUST, left))
        right = max(-self.MAX_THRUST, min(self.MAX_THRUST, right))

        # Publish
        self.left_pub.publish(Float64(data=float(left)))
        self.right_pub.publish(Float64(data=float(right)))

        # Log
        if abs(left) > 1 or abs(right) > 1:
            mode = "OFFBOARD" if self.offboard_active else "DIRECT"
            self.get_logger().info(
                f"[{mode}] vel={self.vel_cmd:.2f} yaw={self.yaw_cmd:.0f} L={left:.0f} R={right:.0f}",
                throttle_duration_sec=0.5
            )

    def mavsdk_thread(self):
        self._mavsdk_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._mavsdk_loop)
        self._mavsdk_loop.run_until_complete(self.mavsdk_main())

    async def mavsdk_main(self):
        try:
            self.get_logger().info("Connecting to PX4...")
            await self.drone.connect(system_address="udp://:14540")

            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    self.get_logger().info("PX4 connected")
                    break

            self.get_logger().info("Waiting for GPS...")
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok:
                    break

            self.get_logger().info("Arming...")
            await self.drone.action.arm()

            self.get_logger().info("Starting offboard...")
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            await self.drone.offboard.start()
            self.offboard_active = True
            self.get_logger().info("OFFBOARD ACTIVE")

            while self.running:
                await asyncio.sleep(0.1)

        except Exception as e:
            self.get_logger().error(f"Offboard error: {e}")
            self.offboard_active = False

def main():
    rclpy.init()
    node = WAMVTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

if __name__ == "__main__":
    main()