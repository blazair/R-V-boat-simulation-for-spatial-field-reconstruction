#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import ExecuteProcess, LogInfo, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    wamv_desc = get_package_share_directory('wamv_description')
    urdf_file = os.path.join(wamv_desc, 'urdf', 'wamv.urdf')
    
    # RViz config
    rviz_config = os.path.join(
        os.path.expanduser('~/karin_ws/src/field_sim/rviz'),
        'field_view.rviz'
    )
    
    with open(urdf_file, 'r') as f:
        robot_desc = f.read()
    
    return LaunchDescription([
        
        LogInfo(msg="Starting Field Validation System..."),
        
        # PX4 + Gazebo
        ExecuteProcess(
            cmd=['make', 'px4_sitl', 'gz_wamv_lake_world'],
            cwd=os.path.expanduser('~/PX4-Autopilot'),
            output='screen',
            name='px4_gazebo'
        ),
        
        TimerAction(
            period=8.0,
            actions=[
                
                # Ground Truth Field
                Node(
                    package='field_sim',
                    executable='ground_truth_field_node.py',
                    name='ground_truth_field',
                    output='screen'
                ),
                
                # GPS Bridge
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    name='gps_bridge',
                    arguments=['/world/lake_world/model/wamv_0/link/base_link/sensor/navsat_sensor/navsat@sensor_msgs/msg/NavSatFix@gz.msgs.NavSat'],
                    output='screen'
                ),
                
                # Odometry Bridge
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    name='odom_bridge',
                    arguments=['/model/wamv_0/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry'],
                    remappings=[('/model/wamv_0/odometry', '/wamv/odom')],
                    output='screen'
                ),
                
                # Thrust bridges
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    name='left_thrust_bridge',
                    arguments=['/wamv_0/command/thrust/left@std_msgs/msg/Float64@gz.msgs.Double'],
                    remappings=[('/wamv_0/command/thrust/left', '/wamv/thrust/left')],
                    output='screen'
                ),
                
                Node(
                    package='ros_gz_bridge',
                    executable='parameter_bridge',
                    name='right_thrust_bridge',
                    arguments=['/wamv_0/command/thrust/right@std_msgs/msg/Float64@gz.msgs.Double'],
                    remappings=[('/wamv_0/command/thrust/right', '/wamv/thrust/right')],
                    output='screen'
                ),
                
                # Robot State Publisher
                Node(
                    package='robot_state_publisher',
                    executable='robot_state_publisher',
                    name='robot_state_publisher',
                    parameters=[{'robot_description': robot_desc}],
                    output='screen'
                ),
                
                # TF Publisher
                Node(
                    package='field_sim',
                    executable='boat_tf_publisher.py',
                    name='boat_tf_publisher',
                    output='screen'
                ),
                    
                # RViz2 with config
                Node(
                    package='rviz2',
                    executable='rviz2',
                    name='rviz2',
                    arguments=['-d', rviz_config],
                    output='screen'
                ),
            ]
        ),
    ])