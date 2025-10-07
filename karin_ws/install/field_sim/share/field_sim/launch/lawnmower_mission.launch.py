from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Temperature sampler
        Node(
            package='field_sim',
            executable='field_sampler.py',
            name='field_sampler',
            output='screen',
            emulate_tty=True,
        ),
        
        # Lawnmower mission
        Node(
            package='field_sim',
            executable='lawnmower_mission.py',
            name='lawnmower_mission',
            output='screen',
            emulate_tty=True,
        ),
    ])