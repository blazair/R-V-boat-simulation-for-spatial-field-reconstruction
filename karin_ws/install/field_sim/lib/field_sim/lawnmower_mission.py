#!/usr/bin/env python3
"""
Lawnmower mission with PD velocity controller for smooth, stable tracking.
Uses direct thrust control with conservative, tuned gains.
"""
import asyncio
import math
import sys
from collections import deque
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed, OffboardError
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import PoseStamped

class LawnmowerMissionPD(Node):
    def __init__(self):
        super().__init__('lawnmower_mission_pd')
        
        # Mission parameters
        self.FIELD_LENGTH = 150.0
        self.FIELD_WIDTH = 42.0
        self.TRACK_SPACING = 5.0
        self.WAYPOINT_TOLERANCE = 2.5
        
        # Control parameters - TUNED FOR STABILITY
        self.MAX_VEL = 0.4           # m/s max cruise velocity
        self.MAX_YAWRATE = 20.0      # deg/s max turn rate
        self.VEL_TO_THRUST = 40.0    # N per m/s (conservative)
        self.YAW_TO_THRUST = 2.0     # N per deg/s
        self.MAX_THRUST = 100.0      # Absolute limit
        
        # PD gains for forward velocity control
        self.KP_VEL = 0.25           # Proportional gain
        self.KD_VEL = 0.15           # Derivative gain (damping)
        
        # PD gains for yaw rate control
        self.KP_YAW = 1.5            # Proportional gain
        self.KD_YAW = 0.3            # Derivative gain (damping)
        
        # Velocity profile parameters
        self.APPROACH_DISTANCE = 10.0  # Start slowing down at this distance
        self.MIN_VEL = 0.1             # Minimum velocity when approaching
        
        # State
        self.current_pose = None
        self.prev_pose = None
        self.start_position = None
        self.waypoints = []
        self.current_wp_idx = 0
        
        # PD controller state
        self.prev_distance_error = 0.0
        self.prev_yaw_error = 0.0
        self.distance_history = deque(maxlen=5)  # For derivative smoothing
        self.yaw_history = deque(maxlen=5)
        self.last_control_time = None
        
        # Velocity smoothing
        self.current_vel_cmd = 0.0
        self.current_yaw_cmd = 0.0
        self.VEL_ACCEL_LIMIT = 0.2    # m/s per control cycle
        self.YAW_ACCEL_LIMIT = 5.0    # deg/s per control cycle
        
        # Subscriber
        self.odom_sub = self.create_subscription(
            Odometry,
            '/wamv/odom',
            self.odom_callback,
            10
        )
        
        # Publishers
        self.left_pub = self.create_publisher(Float64, '/wamv/thrust/left', 10)
        self.right_pub = self.create_publisher(Float64, '/wamv/thrust/right', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/mission/current_target', 10)
        
        # MAVSDK
        self.drone = System()
        
        self.get_logger().info("="*60)
        self.get_logger().info("WAMV Lawnmower Mission - PD CONTROLLER")
        self.get_logger().info(f"Field: {self.FIELD_LENGTH}m × {self.FIELD_WIDTH}m")
        self.get_logger().info(f"Max speed: {self.MAX_VEL} m/s (conservative)")
        self.get_logger().info(f"PD Gains: Kp_vel={self.KP_VEL}, Kd_vel={self.KD_VEL}")
        self.get_logger().info(f"          Kp_yaw={self.KP_YAW}, Kd_yaw={self.KD_YAW}")
        self.get_logger().info("="*60)
        
    def odom_callback(self, msg: Odometry):
        self.prev_pose = self.current_pose
        self.current_pose = msg.pose.pose
    
    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)
    
    def normalize_angle(self, angle):
        """Wrap angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def generate_waypoints(self):
        """
        42m tracks forward, 40m tracks backward (avoids overshooting field)
        """
        waypoints = []
        start_x = self.start_position.position.x
        start_y = self.start_position.position.y
        start_yaw = self.get_yaw_from_quaternion(self.start_position.orientation)

        # After turning left, face short dimension
        track_direction_yaw = start_yaw + math.pi/2

        # Right direction (where we shift between tracks)
        right_yaw = track_direction_yaw - math.pi/2

        # Number of tracks = cover 150m width with 5m spacing
        num_tracks = int(self.FIELD_LENGTH / self.TRACK_SPACING) + 1  # 30 tracks

        self.get_logger().info(f"Generating {num_tracks} tracks (42m forward, 40m back)")

        for i in range(num_tracks):
            right_offset = i * self.TRACK_SPACING

            if i % 2 == 0:  # Even: go FORWARD 42m
                x1 = start_x + right_offset * math.cos(right_yaw)
                y1 = start_y + right_offset * math.sin(right_yaw)
                x2 = x1 + 42.0 * math.cos(track_direction_yaw)
                y2 = y1 + 42.0 * math.sin(track_direction_yaw)
                yaw = track_direction_yaw

            else:  # Odd: go BACKWARD 40m
                x1 = start_x + right_offset * math.cos(right_yaw) + 42.0 * math.cos(track_direction_yaw)
                y1 = start_y + right_offset * math.sin(right_yaw) + 42.0 * math.sin(track_direction_yaw)
                x2 = x1 - 40.0 * math.cos(track_direction_yaw)
                y2 = y1 - 40.0 * math.sin(track_direction_yaw)
                yaw = track_direction_yaw + math.pi

            waypoints.append({'x': x1, 'y': y1, 'yaw': yaw})
            waypoints.append({'x': x2, 'y': y2, 'yaw': yaw})

        self.waypoints = waypoints
        self.get_logger().info(f"Generated {len(waypoints)} waypoints")

        for i in range(min(6, len(waypoints))):
            wp = waypoints[i]
            self.get_logger().info(
                f"  WP{i}: x={wp['x']:.1f}m, y={wp['y']:.1f}m, yaw={math.degrees(wp['yaw']):.0f}°"
            )
    
    def calculate_pd_control(self, target_wp, dt):
        """PD controller for velocity and yaw rate"""
        if self.current_pose is None:
            return 0.0, 0.0, float('inf')
        
        # Current state
        x = self.current_pose.position.x
        y = self.current_pose.position.y
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        
        # Position error
        dx = target_wp['x'] - x
        dy = target_wp['y'] - y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Desired heading
        desired_yaw = math.atan2(dy, dx)
        yaw_error = self.normalize_angle(desired_yaw - current_yaw)
        
        # === Forward Velocity PD Control ===
        
        # Proportional term: based on distance
        distance_error = distance
        P_vel = self.KP_VEL * distance_error
        
        # Derivative term: rate of change of distance error (smoothed)
        self.distance_history.append(distance_error)
        if len(self.distance_history) >= 2 and dt > 0:
            distance_error_rate = (self.distance_history[-1] - self.distance_history[0]) / (dt * len(self.distance_history))
            D_vel = self.KD_VEL * distance_error_rate
        else:
            D_vel = 0.0
        
        # Combined PD output
        vel_cmd = P_vel + D_vel
        
        # Velocity profile: slow down when approaching waypoint
        if distance < self.APPROACH_DISTANCE:
            vel_scale = max(0.3, distance / self.APPROACH_DISTANCE)
            vel_cmd *= vel_scale
        
        # Reduce speed when not aligned with target
        alignment = math.cos(yaw_error)
        if alignment < 0.7:  # More than ~45 degrees off
            vel_cmd *= max(0.3, alignment)
        
        # Clamp velocity
        vel_cmd = max(self.MIN_VEL, min(self.MAX_VEL, vel_cmd))
        
        # === Yaw Rate PD Control ===
        
        # Proportional term
        P_yaw = self.KP_YAW * math.degrees(yaw_error)
        
        # Derivative term: rate of change of yaw error (smoothed)
        self.yaw_history.append(yaw_error)
        if len(self.yaw_history) >= 2 and dt > 0:
            yaw_error_rate = (self.yaw_history[-1] - self.yaw_history[0]) / (dt * len(self.yaw_history))
            D_yaw = self.KD_YAW * math.degrees(yaw_error_rate)
        else:
            D_yaw = 0.0
        
        # Combined PD output
        yaw_rate = P_yaw + D_yaw
        
        # Clamp yaw rate
        yaw_rate = max(-self.MAX_YAWRATE, min(self.MAX_YAWRATE, yaw_rate))
        
        return vel_cmd, yaw_rate, distance
    
    def smooth_commands(self, target_vel, target_yaw):
        """Apply acceleration limits for smooth control"""
        # Smooth velocity
        vel_error = target_vel - self.current_vel_cmd
        if abs(vel_error) > self.VEL_ACCEL_LIMIT:
            self.current_vel_cmd += math.copysign(self.VEL_ACCEL_LIMIT, vel_error)
        else:
            self.current_vel_cmd = target_vel
        
        # Smooth yaw rate
        yaw_error = target_yaw - self.current_yaw_cmd
        if abs(yaw_error) > self.YAW_ACCEL_LIMIT:
            self.current_yaw_cmd += math.copysign(self.YAW_ACCEL_LIMIT, yaw_error)
        else:
            self.current_yaw_cmd = target_yaw
        
        return self.current_vel_cmd, self.current_yaw_cmd
    
    def publish_thrust(self, vel_cmd, yaw_cmd):
        """Convert velocity commands to differential thrust"""
        base_thrust = vel_cmd * self.VEL_TO_THRUST
        yaw_thrust = yaw_cmd * self.YAW_TO_THRUST
        
        left = base_thrust - yaw_thrust
        right = base_thrust + yaw_thrust
        
        # Clamp
        left = max(-self.MAX_THRUST, min(self.MAX_THRUST, left))
        right = max(-self.MAX_THRUST, min(self.MAX_THRUST, right))
        
        self.left_pub.publish(Float64(data=float(left)))
        self.right_pub.publish(Float64(data=float(right)))
        
        return left, right
    
    async def run_mission(self):
        try:
            # Connect
            self.get_logger().info("Connecting to PX4...")
            await self.drone.connect(system_address="udp://:14540")
            
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    self.get_logger().info("✓ PX4 connected")
                    break
            
            # Wait GPS
            self.get_logger().info("Waiting for GPS...")
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok:
                    self.get_logger().info("✓ GPS ready")
                    break
            
            # Wait odometry
            while self.current_pose is None:
                await asyncio.sleep(0.1)
            
            self.start_position = self.current_pose
            self.generate_waypoints()
            
            # Arm
            self.get_logger().info("Arming...")
            await self.drone.action.arm()
            self.get_logger().info("✓ Armed")
            
            await asyncio.sleep(1.0)
            
            # Start offboard
            self.get_logger().info("Starting offboard...")
            await self.drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            await self.drone.offboard.start()
            self.get_logger().info("✓ Offboard active")
            
            await asyncio.sleep(1.0)
            
            # Mission loop
            self.current_wp_idx = 1  # Skip start position
            self.last_control_time = self.get_clock().now()
            
            self.get_logger().info("Starting mission with PD controller...")
            
            while self.current_wp_idx < len(self.waypoints):
                loop_start = self.get_clock().now()
                
                wp = self.waypoints[self.current_wp_idx]
                
                # Calculate dt
                current_time = self.get_clock().now()
                dt = (current_time - self.last_control_time).nanoseconds / 1e9
                self.last_control_time = current_time
                dt = max(0.01, min(0.5, dt))  # Clamp to reasonable range
                
                # PD control
                target_vel, target_yaw, dist = self.calculate_pd_control(wp, dt)
                
                # Smooth commands
                vel_cmd, yaw_cmd = self.smooth_commands(target_vel, target_yaw)
                
                # Send to offboard (telemetry only)
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(vel_cmd, 0.0, 0.0, yaw_cmd)
                )
                
                # Publish thrust (actual control)
                left, right = self.publish_thrust(vel_cmd, yaw_cmd)
                
                # Log progress
                if self.current_wp_idx % 2 == 1:
                    self.get_logger().info(
                        f"→ WP{self.current_wp_idx}/{len(self.waypoints)} | "
                        f"dist={dist:.1f}m | vel={vel_cmd:.2f}m/s | "
                        f"yaw={yaw_cmd:.1f}°/s | L={left:.0f}N R={right:.0f}N",
                        throttle_duration_sec=2.0
                    )
                
                # Check waypoint reached
                if dist < self.WAYPOINT_TOLERANCE:
                    self.get_logger().info(f"✓ WP{self.current_wp_idx} reached (dist={dist:.2f}m)")
                    self.current_wp_idx += 1
                    
                    # Reset controller state at waypoint
                    self.distance_history.clear()
                    self.yaw_history.clear()
                    
                    # Brief pause
                    await asyncio.sleep(1.0)
                
                # Control rate: 20Hz
                await asyncio.sleep(0.05)
            
            # Mission complete
            self.get_logger().info("="*60)
            self.get_logger().info("✓ MISSION COMPLETE")
            self.get_logger().info("="*60)
            
            # Stop smoothly
            for i in range(10):
                scale = 1.0 - (i / 10.0)
                await self.drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(self.current_vel_cmd * scale, 0.0, 0.0, 0.0)
                )
                self.publish_thrust(self.current_vel_cmd * scale, 0.0)
                await asyncio.sleep(0.1)
            
            self.publish_thrust(0.0, 0.0)
            await asyncio.sleep(1.0)
            await self.drone.offboard.stop()
            
        except Exception as e:
            self.get_logger().error(f"Mission error: {e}")
            import traceback
            traceback.print_exc()

def main():
    rclpy.init()
    node = LawnmowerMissionPD()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        async def spin_ros():
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.01)
                await asyncio.sleep(0.01)
        
        loop.run_until_complete(asyncio.gather(
            node.run_mission(),
            spin_ros()
        ))
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()