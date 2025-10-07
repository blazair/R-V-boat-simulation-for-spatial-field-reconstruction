#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import asyncio
import math
from mavsdk import System
from mavsdk.offboard import PositionNedYaw, OffboardError

class AutonomousMission(Node):
    def __init__(self):
        super().__init__('autonomous_mission')
        
        # Field bounds (150m x 50m, starting at origin)
        self.field_bounds = {
            'x_min': 0.0,
            'x_max': 140.0,  # Stay 10m from edge
            'y_min': 5.0,    # Stay 5m from left bank
            'y_max': 45.0    # Stay 5m from right bank
        }
        
        self.drone = System()
        self.get_logger().info('Autonomous Mission Node Started')
        
        # Run mission in separate thread
        import threading
        threading.Thread(target=self.run_mission_thread, daemon=True).start()
    
    def run_mission_thread(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_mission())
    
    async def run_mission(self):
        try:
            self.get_logger().info('Connecting to PX4...')
            await self.drone.connect(system_address="udp://:14540")
            
            async for state in self.drone.core.connection_state():
                if state.is_connected:
                    self.get_logger().info('PX4 Connected!')
                    break
            
            # Wait for GPS
            self.get_logger().info('Waiting for GPS...')
            async for health in self.drone.telemetry.health():
                if health.is_global_position_ok:
                    self.get_logger().info('GPS OK')
                    break
            
            # Get home position
            async for position in self.drone.telemetry.position():
                home_lat = position.latitude_deg
                home_lon = position.longitude_deg
                self.get_logger().info(f'Home: {home_lat}, {home_lon}')
                break
            
            # Arm
            self.get_logger().info('Arming...')
            await self.drone.action.arm()
            
            # Start offboard with initial setpoint
            self.get_logger().info('Starting offboard mode...')
            await self.drone.offboard.set_position_ned(PositionNedYaw(0.0, 0.0, 0.0, 0.0))
            await self.drone.offboard.start()
            
            # Execute lawnmower pattern (one U-turn)
            await self.execute_lawnmower()
            
            # Return to start
            self.get_logger().info('Mission complete, returning to start...')
            await self.goto_position(0.0, 0.0, 0.0)
            
            # Stop offboard
            await self.drone.offboard.stop()
            
            self.get_logger().info('Mission finished!')
            
        except OffboardError as e:
            self.get_logger().error(f'Offboard error: {e}')
        except Exception as e:
            self.get_logger().error(f'Mission error: {e}')
    
    async def execute_lawnmower(self):
        """Execute one pass of lawnmower pattern"""
        
        # Leg 1: Go forward 30m (North in NED)
        self.get_logger().info('Leg 1: Forward 30m')
        await self.goto_position(30.0, 0.0, 0.0)
        await asyncio.sleep(2)
        
        # Leg 2: Move right 15m (East in NED) 
        self.get_logger().info('Leg 2: Right 15m')
        await self.goto_position(30.0, 15.0, 90.0)
        await asyncio.sleep(2)
        
        # Leg 3: Go forward 30m
        self.get_logger().info('Leg 3: Forward 30m')
        await self.goto_position(60.0, 15.0, 0.0)
        await asyncio.sleep(2)
        
        # Leg 4: Move right 15m
        self.get_logger().info('Leg 4: Right 15m')
        await self.goto_position(60.0, 30.0, 90.0)
        await asyncio.sleep(2)
        
        # Leg 5: Return
        self.get_logger().info('Leg 5: Return 60m')
        await self.goto_position(0.0, 30.0, 180.0)
        await asyncio.sleep(2)
    
    async def goto_position(self, north, east, yaw_deg):
        """Go to NED position with yaw"""
        # NED: North, East, Down (down=0 for surface vehicle)
        
        # Check bounds (NED north = our x, east = our y)
        if not (self.field_bounds['x_min'] <= north <= self.field_bounds['x_max']):
            self.get_logger().warn(f'Position {north}m exceeds X bounds!')
            return
        
        if not (self.field_bounds['y_min'] <= east <= self.field_bounds['y_max']):
            self.get_logger().warn(f'Position {east}m exceeds Y bounds!')
            return
        
        self.get_logger().info(f'Going to: N={north:.1f}m, E={east:.1f}m, Yaw={yaw_deg}°')
        
        # Send position setpoint
        await self.drone.offboard.set_position_ned(
            PositionNedYaw(north, east, 0.0, yaw_deg)
        )
        
        # Wait until reached (simple timeout - real version would check position)
        await asyncio.sleep(15)  # Adjust based on boat speed
    
def main():
    rclpy.init()
    node = AutonomousMission()
    rclpy.spin(node)

if __name__ == '__main__':
    main()