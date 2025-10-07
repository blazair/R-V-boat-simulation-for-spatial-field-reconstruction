#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64, ColorRGBA
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry
import numpy as np
import yaml
import os
import pyproj

class GroundTruthFieldNode(Node):
    def __init__(self):
        super().__init__('ground_truth_field_node')
        
        # GPS origin - Tempe Town Lake starting position
        self.origin_lat = 33.430426
        self.origin_lon = -111.928819
        
        # Parameters
        self.declare_parameter('bounds_x', [0.0, 150.0])  # 150m length (North-South)
        self.declare_parameter('bounds_y', [0.0, 50.0])   # 50m width (East-West)
        self.declare_parameter('resolution', 1.0)
        self.declare_parameter('num_gaussian_bumps', 8)
        self.declare_parameter('num_anisotropic_features', 4)
        self.declare_parameter('num_sharp_fronts', 2)
        self.declare_parameter('field_z_height', -1.0)
        self.declare_parameter('save_params', True)
        
        bounds_x = self.get_parameter('bounds_x').value
        bounds_y = self.get_parameter('bounds_y').value
        self.resolution = self.get_parameter('resolution').value
        num_bumps = self.get_parameter('num_gaussian_bumps').value
        num_aniso = self.get_parameter('num_anisotropic_features').value
        num_fronts = self.get_parameter('num_sharp_fronts').value
        self.field_z = self.get_parameter('field_z_height').value
        
        # Setup coordinate transformation (ENU <-> GPS)
        self.setup_coordinate_transform()
        
        # Generate ground truth field
        self.get_logger().info('Generating geotagged field for Tempe Town Lake...')
        self.x_grid = np.arange(bounds_x[0], bounds_x[1], self.resolution)
        self.y_grid = np.arange(bounds_y[0], bounds_y[1], self.resolution)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid)
        
        # Convert grid to GPS coordinates
        self.lat_grid, self.lon_grid = self.enu_to_gps(self.X, self.Y)
        
        # Generate field with known parameters (deterministic)
        np.random.seed(42)
        self.field_params = {
            'origin_lat': self.origin_lat,
            'origin_lon': self.origin_lon,
            'bounds_m': {'x': bounds_x, 'y': bounds_y},
            'background_gradient_x': 0.08,
            'background_gradient_y': 0.05,
            'base_temperature': 20.0,
            'gaussian_bumps': [],
            'anisotropic_features': [],
            'sharp_fronts': []
        }
        
        self.temperature_field = np.zeros_like(self.X)
        
        # Add background gradient
        self.temperature_field += self.field_params['base_temperature']
        self.temperature_field += self.field_params['background_gradient_x'] * (self.X - self.X.min())
        self.temperature_field += self.field_params['background_gradient_y'] * (self.Y - self.Y.min())
        
        # Add isotropic Gaussian bumps
        for i in range(num_bumps):
            cx = np.random.uniform(bounds_x[0] + 15, bounds_x[1] - 15)
            cy = np.random.uniform(bounds_y[0] + 10, bounds_y[1] - 10)
            sigma = np.random.uniform(4, 12)
            amplitude = np.random.uniform(-6, 12)
            
            bump = amplitude * np.exp(-((self.X - cx)**2 + (self.Y - cy)**2) / (2 * sigma**2))
            self.temperature_field += bump
            
            self.field_params['gaussian_bumps'].append({
                'center_x': float(cx),
                'center_y': float(cy),
                'sigma': float(sigma),
                'amplitude': float(amplitude)
            })
        
        # Add ANISOTROPIC features
        for i in range(num_aniso):
            cx = np.random.uniform(bounds_x[0] + 15, bounds_x[1] - 15)
            cy = np.random.uniform(bounds_y[0] + 10, bounds_y[1] - 10)
            sigma_major = np.random.uniform(15, 25)
            sigma_minor = np.random.uniform(3, 8)
            theta = np.random.uniform(0, np.pi)
            amplitude = np.random.uniform(-5, 8)
            
            X_rot = (self.X - cx) * np.cos(theta) + (self.Y - cy) * np.sin(theta)
            Y_rot = -(self.X - cx) * np.sin(theta) + (self.Y - cy) * np.cos(theta)
            
            aniso_feature = amplitude * np.exp(-(X_rot**2 / (2 * sigma_major**2) + 
                                                  Y_rot**2 / (2 * sigma_minor**2)))
            self.temperature_field += aniso_feature
            
            self.field_params['anisotropic_features'].append({
                'center_x': float(cx),
                'center_y': float(cy),
                'sigma_major': float(sigma_major),
                'sigma_minor': float(sigma_minor),
                'theta_rad': float(theta),
                'amplitude': float(amplitude),
                'anisotropy_ratio': float(sigma_major / sigma_minor)
            })
        
        # Add SHARP FRONTS
        for i in range(num_fronts):
            cx = np.random.uniform(bounds_x[0] + 20, bounds_x[1] - 20)
            cy = np.random.uniform(bounds_y[0] + 10, bounds_y[1] - 10)
            theta = np.random.uniform(0, np.pi)
            amplitude = np.random.uniform(4, 8)
            sharpness = np.random.uniform(0.1, 0.3)
            
            d = (self.X - cx) * np.cos(theta) + (self.Y - cy) * np.sin(theta)
            front = amplitude * np.tanh(d * sharpness)
            self.temperature_field += front
            
            self.field_params['sharp_fronts'].append({
                'center_x': float(cx),
                'center_y': float(cy),
                'theta_rad': float(theta),
                'amplitude': float(amplitude),
                'sharpness': float(sharpness)
            })
        
        # Add spatially-varying noise
        noise_base = 0.3
        noise_variation = 0.4 * np.exp(-((self.X - np.mean(self.X))**2 + 
                                         (self.Y - np.mean(self.Y))**2) / (30**2))
        noise = np.random.normal(0, noise_base + noise_variation, self.temperature_field.shape)
        self.temperature_field += noise
        
        if self.get_parameter('save_params').value:
            self._save_field_params()
        
        # Publishers
        self.field_viz_pub = self.create_publisher(
            MarkerArray, '/field/ground_truth/visualization', 10)

        # Service to sample field at boat's current position
        self.sample_service = self.create_service(
            Trigger,
            '/sample_field',
            self.sample_field_callback
        )

        # Subscribe to boat position for sampling
        self.current_boat_pose = None
        self.boat_sub = self.create_subscription(
            Odometry,
            '/wamv/odom',
            self.boat_pose_callback,
            10
        )

        self.get_logger().info("✓ /sample_field service created")

        self.viz_timer = self.create_timer(1.0, self.publish_visualization)
        
        self.temp_min = float(np.min(self.temperature_field))
        self.temp_max = float(np.max(self.temperature_field))
        
        self.get_logger().info(f'Field generated: {self.X.shape[0]}x{self.X.shape[1]} grid')
        self.get_logger().info(f'Field dimensions: {bounds_x[1]-bounds_x[0]}m x {bounds_y[1]-bounds_y[0]}m')
        self.get_logger().info(f'GPS origin: ({self.origin_lat}, {self.origin_lon})')
        self.get_logger().info(f'Temperature range: [{self.temp_min:.2f}, {self.temp_max:.2f}] °C')
        
    def setup_coordinate_transform(self):
        """Setup ENU (East-North-Up) to GPS coordinate transformation"""
        # WGS84 (GPS coords)
        self.wgs84 = pyproj.CRS('EPSG:4326')
        
        # Local ENU frame centered at origin
        self.enu = pyproj.Proj(proj='aeqd', lat_0=self.origin_lat, 
                               lon_0=self.origin_lon, datum='WGS84')
        
        self.transformer_to_gps = pyproj.Transformer.from_proj(
            self.enu, self.wgs84, always_xy=True)
        self.transformer_to_enu = pyproj.Transformer.from_proj(
            self.wgs84, self.enu, always_xy=True)
    
    def enu_to_gps(self, x, y):
        """Convert ENU (x=East, y=North) to GPS (lat, lon)"""
        lon, lat = self.transformer_to_gps.transform(x, y)
        return lat, lon
    
    def gps_to_enu(self, lat, lon):
        """Convert GPS (lat, lon) to ENU (x=East, y=North)"""
        x, y = self.transformer_to_enu.transform(lon, lat)
        return x, y
        
    def _save_field_params(self):
        """Save field parameters to YAML"""
        package_share = os.path.join(os.path.expanduser('~'), 'karin_ws', 'src', 'field_sim', 'config')
        os.makedirs(package_share, exist_ok=True)
        param_file = os.path.join(package_share, 'generated_field_params.yaml')
        
        with open(param_file, 'w') as f:
            yaml.dump(self.field_params, f, default_flow_style=False)
        
        self.get_logger().info(f'Field parameters saved to: {param_file}')
    
    def sample_field_gps(self, lat, lon):
        """Sample field at GPS coordinates, return (temperature, x, y, lat, lon)"""
        # Convert GPS to ENU
        x, y = self.gps_to_enu(lat, lon)
        
        # Check bounds
        if x < self.x_grid[0] or x > self.x_grid[-1] or \
           y < self.y_grid[0] or y > self.y_grid[-1]:
            return None
        
        # Bilinear interpolation
        i = np.searchsorted(self.x_grid, x) - 1
        j = np.searchsorted(self.y_grid, y) - 1
        
        if i < 0 or i >= len(self.x_grid)-1 or j < 0 or j >= len(self.y_grid)-1:
            return None
        
        x0, x1 = self.x_grid[i], self.x_grid[i+1]
        y0, y1 = self.y_grid[j], self.y_grid[j+1]
        
        wx = (x - x0) / (x1 - x0)
        wy = (y - y0) / (y1 - y0)
        
        temp = (1-wx)*(1-wy)*self.temperature_field[j,i] + \
               wx*(1-wy)*self.temperature_field[j,i+1] + \
               (1-wx)*wy*self.temperature_field[j+1,i] + \
               wx*wy*self.temperature_field[j+1,i+1]
        
        return {
            'temperature': float(temp),
            'x_enu': float(x),
            'y_enu': float(y),
            'latitude': float(lat),
            'longitude': float(lon)
        }
    
    def publish_visualization(self):
        """Publish field as MarkerArray"""
        marker_array = MarkerArray()
        
        marker_id = 0
        step = max(1, int(len(self.x_grid) / 60))
        
        for i in range(0, len(self.x_grid), step):
            for j in range(0, len(self.y_grid), step):
                marker = Marker()
                marker.header.frame_id = "wamv_0/odom"  # CHANGED from "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "ground_truth_field"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                marker.pose.position.x = float(self.x_grid[i])
                marker.pose.position.y = float(self.y_grid[j])
                marker.pose.position.z = self.field_z
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = self.resolution * step
                marker.scale.y = self.resolution * step
                marker.scale.z = 0.3
                
                temp = self.temperature_field[j, i]
                normalized = (temp - self.temp_min) / (self.temp_max - self.temp_min)
                
                color = self._temperature_to_color(normalized)
                marker.color = color
                
                marker_array.markers.append(marker)
                marker_id += 1
        
        self.field_viz_pub.publish(marker_array)
    
    def _temperature_to_color(self, normalized_value):
        """Convert normalized temperature to color"""
        color = ColorRGBA()
        color.a = 0.75
        
        if normalized_value < 0.25:
            t = normalized_value / 0.25
            color.r = 0.0
            color.g = t * 0.5
            color.b = 1.0
        elif normalized_value < 0.5:
            t = (normalized_value - 0.25) / 0.25
            color.r = 0.0
            color.g = 0.5 + t * 0.5
            color.b = 1.0 - t
        elif normalized_value < 0.75:
            t = (normalized_value - 0.5) / 0.25
            color.r = t
            color.g = 1.0
            color.b = 0.0
        else:
            t = (normalized_value - 0.75) / 0.25
            color.r = 1.0
            color.g = 1.0 - t * 0.5
            color.b = 0.0
        
        return color

    def boat_pose_callback(self, msg):
        """Store current boat position"""
        self.current_boat_pose = msg.pose.pose

    def sample_field_callback(self, request, response):
        """Service callback to sample field at boat's current position"""
        try:
            if self.current_boat_pose is None:
                response.success = False
                response.message = "No boat position available"
                return response
            
            # Get boat position in ENU
            x = self.current_boat_pose.position.x
            y = self.current_boat_pose.position.y
            
            # Convert to GPS
            lat, lon = self.enu_to_gps(x, y)
            
            # Sample field
            result = self.sample_field_gps(lat, lon)
            
            if result is None:
                response.success = False
                response.message = "Position out of bounds"
            else:
                response.success = True
                # Return temperature as string in the message field
                response.message = str(result['temperature'])
            
            return response
            
        except Exception as e:
            response.success = False
            response.message = f"Error: {str(e)}"
            return response

def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthFieldNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()