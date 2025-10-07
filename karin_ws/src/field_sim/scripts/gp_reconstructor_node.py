#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GPReconstructorNode(Node):
    def __init__(self):
        super().__init__('gp_reconstructor_node')
        
        self.samples_x = []
        self.samples_y = []
        self.samples_temp = []
        
        # Setup matplotlib figure (separate window)
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle('GP Reconstruction vs Ground Truth')
        
        # Timer to update plot
        self.plot_timer = self.create_timer(5.0, self.update_plot)
        
        # Timer to rebuild GP when we have enough samples
        self.gp_timer = self.create_timer(10.0, self.rebuild_gp)
        
    def rebuild_gp(self):
        if len(self.samples_x) < 10:
            return
        
        self.get_logger().info(f'Rebuilding GP with {len(self.samples_x)} samples')
        
        # TODO: Call your NS-KNN GP code here
        # For now, placeholder
        
    def update_plot(self):
        """Update matplotlib window with current state"""
        self.ax1.clear()
        self.ax2.clear()
        
        # Left plot: sample locations
        if len(self.samples_x) > 0:
            self.ax1.scatter(self.samples_x, self.samples_y, 
                           c=self.samples_temp, cmap='viridis', s=50)
            self.ax1.set_title(f'Samples Collected: {len(self.samples_x)}')
        
        # Right plot: GP reconstruction (when ready)
        self.ax2.set_title('GP Reconstruction (updating...)')
        
        plt.pause(0.01)

def main():
    rclpy.init()
    node = GPReconstructorNode()
    
    plt.ion()  # Interactive mode
    plt.show()
    
    rclpy.spin(node)

if __name__ == '__main__':
    main()