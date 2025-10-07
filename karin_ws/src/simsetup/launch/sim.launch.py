#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os

def generate_launch_description():
    
    return LaunchDescription([
        
        LogInfo(msg="Starting WAM-V PX4 Simulation with Gazebo-ROS2 Bridge"),
        
        # Launch PX4 SITL
        ExecuteProcess(
            cmd=['make', 'px4_sitl', 'gz_wamv_lake_world'],
            cwd=os.path.expanduser('~/PX4-Autopilot'),
            output='screen',
            name='px4_sitl'
        ),
        
        # Bridge: Clock
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='clock_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
            output='screen'
        ),
        
        # Bridge: IMU
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='imu_bridge',
            arguments=[
                '/world/lake_world/model/wamv_0/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
            ],
            remappings=[
                ('/world/lake_world/model/wamv_0/link/base_link/sensor/imu_sensor/imu', '/wamv/imu')
            ],
            output='screen'
        ),
        
        # Bridge: NavSat (GPS)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='navsat_bridge',
            arguments=[
                '/world/lake_world/model/wamv_0/link/base_link/sensor/navsat_sensor/navsat@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat'
            ],
            remappings=[
                ('/world/lake_world/model/wamv_0/link/base_link/sensor/navsat_sensor/navsat', '/wamv/gps')
            ],
            output='screen'
        ),
        
        # Bridge: Magnetometer
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='mag_bridge',
            arguments=[
                '/world/lake_world/model/wamv_0/link/base_link/sensor/magnetometer_sensor/magnetometer@sensor_msgs/msg/MagneticField[gz.msgs.Magnetometer'
            ],
            remappings=[
                ('/world/lake_world/model/wamv_0/link/base_link/sensor/magnetometer_sensor/magnetometer', '/wamv/mag')
            ],
            output='screen'
        ),
        
        # Bridge: Air Pressure
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='air_pressure_bridge',
            arguments=[
                '/world/lake_world/model/wamv_0/link/base_link/sensor/air_pressure_sensor/air_pressure@sensor_msgs/msg/FluidPressure[gz.msgs.FluidPressure'
            ],
            remappings=[
                ('/world/lake_world/model/wamv_0/link/base_link/sensor/air_pressure_sensor/air_pressure', '/wamv/air_pressure')
            ],
            output='screen'
        ),
        
        # Bridge: Odometry
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='odom_bridge',
            arguments=[
                '/model/wamv_0/odometry_with_covariance@nav_msgs/msg/Odometry[gz.msgs.OdometryWithCovariance'
            ],
            remappings=[
                ('/model/wamv_0/odometry_with_covariance', '/wamv/odom')
            ],
            output='screen'
        ),
        
        # Bridge: Pose
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='pose_bridge',
            arguments=[
                '/world/lake_world/dynamic_pose/info@geometry_msgs/msg/PoseArray[gz.msgs.Pose_V'
            ],
            remappings=[
                ('/world/lake_world/dynamic_pose/info', '/wamv/pose')
            ],
            output='screen'
        ),
        
        # Bridge: Left Thruster Command (bidirectional)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='left_thrust_bridge',
            arguments=[
                '/wamv_0/command/thrust/left@std_msgs/msg/Float64]gz.msgs.Double'
            ],
            remappings=[
                ('/wamv_0/command/thrust/left', '/wamv/thrust/left')
            ],
            output='screen'
        ),
        
        # Bridge: Right Thruster Command (bidirectional)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='right_thrust_bridge',
            arguments=[
                '/wamv_0/command/thrust/right@std_msgs/msg/Float64]gz.msgs.Double'
            ],
            remappings=[
                ('/wamv_0/command/thrust/right', '/wamv/thrust/right')
            ],
            output='screen'
        ),
        
        # Bridge: Left Thruster Angular Velocity (feedback)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='left_ang_vel_bridge',
            arguments=[
                '/model/wamv_0/joint/left_propeller_joint/ang_vel@std_msgs/msg/Float64[gz.msgs.Double'
            ],
            remappings=[
                ('/model/wamv_0/joint/left_propeller_joint/ang_vel',  '/wamv/thrust/left/ang_vel')
            ],
            output='screen'
        ),
        
        # Bridge: Right Thruster Angular Velocity (feedback)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='right_ang_vel_bridge',
            arguments=[
                '/model/wamv_0/joint/right_propeller_joint/ang_vel@std_msgs/msg/Float64[gz.msgs.Double'
            ],
            remappings=[
                ('/model/wamv_0/joint/right_propeller_joint/ang_vel', '/wamv/thrust/right/ang_vel')
            ],
            output='screen'
        ),
        
        # Echo all bridged topics
        LogInfo(msg="Gazebo-ROS2 Bridge Active! Topics available:"),
        LogInfo(msg="  Sensors: /wamv/imu, /wamv/gps, /wamv/mag, /wamv/air_pressure"),
        LogInfo(msg="  State: /wamv/odom, /wamv/pose"),
        LogInfo(msg="  Control: /wamv/thrust/left, /wamv/thrust/right"),
        LogInfo(msg="  Feedback: /wamv/thrust/left/ang_vel, /wamv/thrust/right/ang_vel"),
        
    ])