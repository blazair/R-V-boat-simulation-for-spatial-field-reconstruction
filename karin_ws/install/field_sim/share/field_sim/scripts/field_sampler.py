#!/usr/bin/env python3
"""
Temperature field sampler with debug output
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
from visualization_msgs.msg import Marker

class FieldSampler(Node):
    def __init__(self):
        super().__init__('field_sampler')
        
        self.SAMPLE_INTERVAL = 2.0
        self.current_pose = None
        self.current_gps = None
        self.last_sample_pos = None
        self.samples_collected = 0
        self.service_ready = False
        
        # CSV setup
        output_dir = Path('/home/blazar/karin_ws/src/simsetup/scripts/data/sim_data')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_filename = output_dir / f'lawnmower_samples_{timestamp}.csv'
        
        self.csv_file = open(self.csv_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'sample_id', 'timestamp', 'latitude', 'longitude',
            'x_meters', 'y_meters', 'temperature_celsius'
        ])
        self.csv_file.flush()
        
        self.get_logger().info(f"CSV created: {self.csv_filename}")
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/wamv/odom', self.odom_callback, 10
        )
        
        self.gps_sub = self.create_subscription(
            NavSatFix,
            '/world/lake_world/model/wamv_0/link/base_link/sensor/navsat_sensor/navsat',
            self.gps_callback, 10
        )
        
        self.marker_pub = self.create_publisher(Marker, '/field/samples/markers', 10)
        
        # Service client
        self.field_client = self.create_client(Trigger, '/sample_field')
        
        # Check service every second
        self.service_check_timer = self.create_timer(1.0, self.check_service)
        self.sample_timer = None
        
        self.get_logger().info("="*60)
        self.get_logger().info("Field Sampler Starting (DEBUG MODE)")
        self.get_logger().info("="*60)
    
    def check_service(self):
        if not self.service_ready:
            if self.field_client.service_is_ready():
                self.service_ready = True
                self.service_check_timer.cancel()
                self.get_logger().info("✓✓✓ SERVICE CONNECTED ✓✓✓")
                # Start sampling timer
                self.sample_timer = self.create_timer(0.5, self.check_and_sample)
                self.get_logger().info("✓✓✓ SAMPLING TIMER STARTED ✓✓✓")
            else:
                self.get_logger().info("Waiting for /sample_field...", throttle_duration_sec=3.0)
    
    def odom_callback(self, msg):
        if self.current_pose is None:
            self.get_logger().info(f"✓ First odometry received: x={msg.pose.pose.position.x:.2f}, y={msg.pose.pose.position.y:.2f}")
        self.current_pose = msg.pose.pose
    
    def gps_callback(self, msg):
        if self.current_gps is None:
            self.get_logger().info(f"✓ First GPS received: lat={msg.latitude:.6f}, lon={msg.longitude:.6f}")
        self.current_gps = msg
    
    def distance_since_last_sample(self):
        if self.last_sample_pos is None or self.current_pose is None:
            return float('inf')
        
        dx = self.current_pose.position.x - self.last_sample_pos.x
        dy = self.current_pose.position.y - self.last_sample_pos.y
        return math.sqrt(dx*dx + dy*dy)
    
    def check_and_sample(self):
        if not self.service_ready:
            self.get_logger().warn("check_and_sample called but service not ready!")
            return
            
        if self.current_pose is None:
            self.get_logger().info("Waiting for odometry...", throttle_duration_sec=2.0)
            return
        
        dist = self.distance_since_last_sample()
        
        # DEBUG: Show distance every few seconds
        self.get_logger().info(
            f"Distance check: {dist:.2f}m (need {self.SAMPLE_INTERVAL}m)",
            throttle_duration_sec=2.0
        )
        
        if dist >= self.SAMPLE_INTERVAL:
            self.get_logger().info(f"✓ Distance threshold reached! Sampling now...")
            self.sample_field()
    
    def sample_field(self):
        try:
            self.get_logger().info("Calling /sample_field service...")
            request = Trigger.Request()
            future = self.field_client.call_async(request)
            future.add_done_callback(self.sample_callback)
        except Exception as e:
            self.get_logger().error(f"Sample error: {e}")
    
    def sample_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Service response: {response.message}")
            temperature = float(response.message)
            self.record_sample(temperature)
        except Exception as e:
            self.get_logger().error(f"Callback error: {e}")
    
    def record_sample(self, temperature):
        pos = self.current_pose.position
        self.last_sample_pos = pos
        self.samples_collected += 1
        
        # Write to CSV
        self.csv_writer.writerow([
            self.samples_collected,
            datetime.now().isoformat(),
            self.current_gps.latitude if self.current_gps else 0.0,
            self.current_gps.longitude if self.current_gps else 0.0,
            f"{pos.x:.6f}",
            f"{pos.y:.6f}",
            f"{temperature:.3f}"
        ])
        self.csv_file.flush()
        
        self.get_logger().info(
            f"✓✓✓ SAMPLE #{self.samples_collected} RECORDED ✓✓✓ "
            f"pos=({pos.x:.2f}, {pos.y:.2f}) temp={temperature:.2f}°C"
        )
        
        self.publish_marker(pos, temperature)
    
    def publish_marker(self, position, temperature):
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "wamv_0/odom"
        marker.ns = "samples"
        marker.id = self.samples_collected
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position = position
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.6
        marker.scale.y = 0.6
        marker.scale.z = 0.6
        
        temp_norm = (temperature - 15.0) / 20.0
        temp_norm = max(0.0, min(1.0, temp_norm))
        
        marker.color.r = temp_norm
        marker.color.g = 0.0
        marker.color.b = 1.0 - temp_norm
        marker.color.a = 1.0
        
        marker.lifetime.sec = 0
        self.marker_pub.publish(marker)
    
    def __del__(self):
        if hasattr(self, 'csv_file') and self.csv_file is not None:
            self.csv_file.close()
            self.get_logger().info(f"✓ CSV closed: {self.samples_collected} samples")

def main():
    rclpy.init()
    node = FieldSampler()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(f"Stopped. {node.samples_collected} samples collected")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()