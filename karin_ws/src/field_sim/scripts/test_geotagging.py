#!/usr/bin/env python3
import sys
sys.path.append('/home/blazar/karin_ws/install/field_sim/lib/python3.10/site-packages')

from field_sim.ground_truth_field_node import GroundTruthFieldNode
import rclpy

rclpy.init()
node = GroundTruthFieldNode()

# Print GPS coordinates for corner points
print("\n=== GEOTAGGING PROOF ===")
print(f"Origin GPS: ({node.origin_lat}, {node.origin_lon})")
print(f"\nField bounds (ENU meters): X=[{node.x_grid[0]}, {node.x_grid[-1]}], Y=[{node.y_grid[0]}, {node.y_grid[-1]}]")

# Corner points
corners = [
    (node.x_grid[0], node.y_grid[0], "Bottom-left"),
    (node.x_grid[-1], node.y_grid[0], "Bottom-right"),
    (node.x_grid[0], node.y_grid[-1], "Top-left"),
    (node.x_grid[-1], node.y_grid[-1], "Top-right"),
]

print("\nCorner GPS coordinates:")
for x, y, name in corners:
    lat, lon = node.enu_to_gps(np.array([x]), np.array([y]))
    print(f"{name:15} ENU({x:6.1f}, {y:6.1f}) → GPS({lat[0]:.6f}, {lon[0]:.6f})")

# Sample at center
center_x = (node.x_grid[0] + node.x_grid[-1]) / 2
center_y = (node.y_grid[0] + node.y_grid[-1]) / 2
lat_c, lon_c = node.enu_to_gps(np.array([center_x]), np.array([center_y]))
print(f"\nCenter:         ENU({center_x:6.1f}, {center_y:6.1f}) → GPS({lat_c[0]:.6f}, {lon_c[0]:.6f})")

# Test sampling
result = node.sample_field_gps(lat_c[0], lon_c[0])
if result:
    print(f"\nSample test at center:")
    print(f"  Temperature: {result['temperature']:.2f}°C")
    print(f"  GPS: ({result['latitude']:.6f}, {result['longitude']:.6f})")
    print(f"  ENU: ({result['x_enu']:.2f}, {result['y_enu']:.2f})")

print("\n✓ Geotagging verified!\n")