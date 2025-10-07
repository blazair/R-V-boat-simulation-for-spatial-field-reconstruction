#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster

class BoatTFPublisher(Node):
    def __init__(self):
        super().__init__('boat_tf_publisher')
        
        self.tf_broadcaster = TransformBroadcaster(self)
        
        self.odom_sub = self.create_subscription(
            Odometry,
            '/wamv/odom',
            self.odom_callback,
            10
        )
        
        self.get_logger().info('TF Publisher waiting for /wamv/odom...')
        
    def odom_callback(self, msg):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = msg.header.frame_id
        t.child_frame_id = 'base_link'
        
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        
        self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = BoatTFPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()