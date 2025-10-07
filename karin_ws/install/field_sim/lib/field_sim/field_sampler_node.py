#!/usr/bin/env python3
"""
Samples temperature field at regular distance intervals during mission.
Calls /sample_field service and logs GPS + temperature to CSV.
"""
import csv
import math
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

class TemperatureSampler(Node):
    def __init__(self):
        super().__init__('temperature_sampler')
        
        # Sampling parameters
        self.SAMPLE_INTERVAL = 2.0  # meters between samples
        self.MIN_SPEED = 0.05       # m/s - only sample when moving
        
        # State
        self.current_pose = None
        self.current_gps = None
        self.last_sample_pos = None
        self.samples_collected = 0
        
        # CSV setup
        output_dir = Path('/home/blazar/karin_ws/src/simsetup/scripts/data/sim_data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = output_dir / f'lawnmower_samples_{timestamp}.csv'
        
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'sample_id',
                'timestamp',
                'latitude',
                'longitude',
                'x_meters',
                'y_meters',
                'temperature_celsius'
            ])
        
        self.get_logger().info(f"Saving to: {self.csv_filename}")
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/wamv/odom',
            self.odom_callback,
            10
        )
        
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/world/lake_world/model/wamv_0/link/base_link/sensor/navsat_sensor/navsat',
            self.gps_callback,
            10
        )
        
        # Publisher for sample markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/field/samples/markers',
            10
        )
        
        # Service client for temperature
        self.field_client = self.create_client(Trigger, '/sample_field')
        
        self.get_logger().info("Waiting for /sample_field service...")
        while not self.field_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Still waiting...")
        self.get_logger().info("✓ Connected to field service")
        
        # Timer for sampling check
        self.timer = self.create_timer(0.5, self.check_and_sample)
        
        self.get_logger().info("="*60)
        self.get_logger().info("Temperature Sampler Active")
        self.get_logger().info(f"Sample interval: {self.SAMPLE_INTERVAL}m")
        self.get_logger().info("="*60)
    
    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose
    
    def gps_callback(self, msg):
        self.current_gps = msg
    
    def distance_since_last_sample(self):
        if self.last_sample_pos is None or self.current_pose is None:
            return float('inf')
        
        dx = self.current_pose.position.x - self.last_sample_pos.x
        dy = self.current_pose.position.y - self.last_sample_pos.y
        return math.sqrt(dx*dx + dy*dy)
    
    def check_and_sample(self):
        if self.current_pose is None:
            return
        
        dist = self.distance_since_last_sample()
        
        if dist >= self.SAMPLE_INTERVAL:
            self.sample_field()
    
    def sample_field(self):
        if self.current_pose is None:
            return
        
        try:
            # Call temperature field service
            request = Trigger.Request()
            future = self.field_client.call_async(request)
            
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            
            if future.result() is not None:
                response = future.result()
                
                try:
                    # Parse temperature from response
                    temperature = float(response.message)
                except ValueError:
                    self.get_logger().warn(f"Could not parse temp: {response.message}")
                    return
                
                self.record_sample(temperature)
            else:
                self.get_logger().warn("Field service call failed")
                
        except Exception as e:
            self.get_logger().error(f"Sampling error: {e}")
    
    def record_sample(self, temperature):
        pos = self.current_pose.position
        
        # Update last sample position
        self.last_sample_pos = pos
        self.samples_collected += 1
        
        # Write to CSV
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.samples_collected,
                datetime.now().isoformat(),
                self.current_gps.latitude if self.current_gps else 0.0,
                self.current_gps.longitude if self.current_gps else 0.0,
                f"{pos.x:.6f}",
                f"{pos.y:.6f}",
                f"{temperature:.3f}"
            ])
        
        # Log
        self.get_logger().info(
            f"Sample #{self.samples_collected}: "
            f"pos=({pos.x:.2f}, {pos.y:.2f}) "
            f"temp={temperature:.2f}°C"
        )
        
        # Visualize
        self.publish_marker(pos, temperature)
    
    def publish_marker(self, position, temperature):
        marker_array = MarkerArray()
        
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "wamv_0/odom"
        marker.ns = "temperature_samples"
        marker.id = self.samples_collected
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position = position
        marker.pose.position.z = 1.0  # Above water
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.8
        marker.scale.y = 0.8
        marker.scale.z = 0.8
        
        # Color: blue (cold) to red (hot), assume 15-35°C range
        temp_norm = (temperature - 15.0) / 20.0
        temp_norm = max(0.0, min(1.0, temp_norm))
        
        marker.color.r = temp_norm
        marker.color.g = 0.0
        marker.color.b = 1.0 - temp_norm
        marker.color.a = 1.0
        
        marker.lifetime.sec = 0  # Permanent
        
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = TemperatureSampler()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f"Stopped. Collected {node.samples_collected} samples")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()