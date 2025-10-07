
#  Spatial Field Reconstruction & Autonomous Sampling Simulation

### DREAMS Lab, Arizona State University  
**Maintainer:** *Bharath Vedantha Desikan*  

---

## Overview

This repository contains a **complete simulation framework** for spatial field reconstruction and Gaussian Process (GP)–based adaptive sampling using a research vessel operating in an aquatic environment.  
The framework integrates **ROS 2 (Jazzy)**, **Gazebo Harmonic**, and a **Paciorek–Schervish Non-Stationary Gaussian Process (KNN–PS GP)** implemented in **CUDA-accelerated Python** for real-time field estimation.

The environment models the behaviour of a robotic surface vehicle navigating over a spatially varying scalar field (e.g. temperature, salinity).  
Sampling, mapping, and reconstruction are performed live through the GP backend, producing quantitative metrics and uncertainty maps.

---

## System Requirements

| Component | Recommended | Notes |
|------------|--------------|-------|
| **OS** | Ubuntu 24.04 LTS |
| **ROS 2** | Jazzy Jalisco |
| **Simulator** | Gazebo Harmonic |
| **GPU** | Mid-tier NVIDIA GPU | CUDA ≥ 12.0 strongly recommended |
| **CPU Fallback** | Any quad-core x86 ≥ 3 GHz | Automatically used if no GPU detected |

When CUDA is available, all GP computations—Cholesky factorisation, covariance construction, and posterior prediction—execute entirely on the GPU.  
If no CUDA device is found, the system automatically falls back to CPU execution while maintaining numerical equivalence (at highly reduced speed).

---


## Installation


### Clone repository
```
git clone https://github.com/blazair/R-V-boat-simulation-for-spatial-field-reconstruction.git
cd R-V-boat-simulation-for-spatial-field-reconstruction/karin_ws
```
### Build workspace
```
colcon build 
source install/setup.bash
```
### Install Python dependencies using the provided requirements file:
```
pip install -r requirements.txt
```

---

>  **Note:** If GPU acceleration is required, install the appropriate CUDA-enabled PyTorch wheel matching your NVIDIA driver and CUDA runtime version, the requirements file have it commented

---

## Running the Simulation

Add the necessary files inside their respective folders in the PX4 directory 


### Test the Simulation:

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_wamv_lake_world
```

### Launch the Environment

```bash
ros2 launch field_sim complete_mission.launch.py 
```

Spawns the research vessel in **Gazebo Harmonic**, loads `lake_world.sdf`, and establishes ROS 2 topic bridges.
Once initialised, the `/sample_field` service becomes available and the boat starts a lawnmower pattern saving CSV file (change the path in the required codes for save and read locations)

---


### Perform Field Reconstruction

After sampling, reconstruct the field using the **KNN–PS GP**:

```bash
python3 scripts/standalone_online_GP_updater.py
```


## CUDA vs CPU Execution

Runtime device selection is handled automatically:

```python
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
TORCH_DTYPE = torch.float64 if (not USE_CUDA or torch.cuda.get_device_capability()[0] >= 7) else torch.float32
```

**When GPU detected:**

* Covariance and Cholesky steps run on CUDA
* Typical runtime ≈ 1 s for a 7 500-point grid

**When no GPU found:**

* Falls back to NumPy/SciPy CPU kernels
* Very high runtime

---


```
