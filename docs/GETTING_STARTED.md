# Getting Started with ManiPylator

ManiPylator is a Python library that provides a unified interface to both analytical robotics (via Robotics Toolbox) and physics-based simulation (via Genesis). Whether you're a student learning robotics, a researcher exploring new algorithms, or a hobbyist building robots, ManiPylator makes it easy to get started.

## Quick Start

### 1. Installation

**Option A: Using Docker (Recommended)**
```bash
# Clone the repository
git clone https://github.com/your-username/manipylator.git
cd manipylator

xhost +local:root # Allow the container to access the display

# Start the Jupyter Lab environment
docker compose up -d

# Open your browser to http://localhost:8888
```

**Option B: Minimal (local development)**

If you only need the MQTT broker and task queue -- for example to work on the
visualizer, state viewer, or any MQTT-driven component -- use the `minimal`
profile.  This skips Klipper, Moonraker, and firmware simulation entirely:

```bash
docker compose up lab -d && docker compose --profile minimal up -d
# Starts: lab (Jupyter, GPU), Mosquitto (port 1883), Redis (port 6379)
```

You can then run the visualizer inside the lab container and push mock state
to drive it:

```bash
docker exec manipylator-lab python /workspace/run_mq_visualizer.py --broker mq

# from another terminal, publish joint state
mosquitto_pub -h localhost -t manipylator/state \
  -m '{"q1": 0.5, "q2": 0.3, "q3": 0.0, "q4": 0.0, "q5": 0.0, "q6": 0.0}'
```

See the [Architecture docs](ARCHITECTURE.md#minimal-profile) for more
examples (mocking devices, subscribing, Panel dashboard).

**Option C: Local Installation**
```bash
# Install dependencies
pip install manipylator

# Or install from source
git clone https://github.com/your-username/manipylator.git
cd manipylator
pip install -e .
```

### 2. Your First Robot

Open the `00-start-here.ipynb` notebook and run:

```python
from manipylator import HeadlessSimulatedRobot
from manipylator.utils import render_robot_from_template

# Load a robot
with render_robot_from_template("robots/empiric") as robot_urdf:
    robot = HeadlessSimulatedRobot(robot_urdf)

# Set joint angles
pose = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# Get end effector position
translation, rotation = robot.step_to_pose(pose)
print(f"End effector position: {translation}")
```

### 3. What You Can Do

- **Forward Kinematics**: Calculate where the robot's end effector is for given joint angles
- **Inverse Kinematics**: Find joint angles to reach a desired position
- **Trajectory Planning**: Generate smooth paths for the robot to follow
- **Simulation**: Visualize robot movements in 3D
- **Real Robot Control**: Control physical robots via MQTT

## Learning Paths

### For Beginners
1. **Start Here**: `00-start-here.ipynb` - Basic concepts and sanity check
2. **Visual Simulation**: `20-simulation.ipynb` - See your robot move in 3D
3. **Symbolic Math**: `10-symbolic-manipulation.ipynb` - Learn the math behind robotics

### For Students
- **Robotics Fundamentals**: Work through the symbolic manipulation notebooks (1X series)
- **Control Theory**: Explore trajectory generation and differential kinematics
- **Practical Implementation**: Try controlling real robots with MQTT

### For Researchers
- **Analytical Methods**: Use Robotics Toolbox for symbolic calculations
- **Physics Simulation**: Leverage Genesis for realistic physics
- **Custom Robots**: Add your own URDF models to the `robots/` directory

### For Hobbyists
- **Hardware Control**: Check out the MQTT examples for real robot control
- **Custom Movements**: Create your own movement sequences
- **Video Recording**: Capture simulations for documentation

## Key Concepts

### Two Interfaces, One Library

ManiPylator provides access to two powerful robotics libraries:

- **Robotics Toolbox (RTB)**: Fast analytical calculations, symbolic math, ideal for algorithms
- **Genesis**: Physics-based simulation, 3D visualization, realistic robot behavior

### Robot Models

Robots are defined using URDF (Universal Robot Description Format) files:
- **Template System**: Use Jinja2 templates for flexible robot configurations
- **Asset Management**: STL files for 3D geometry
- **Easy Loading**: Simple context managers handle file management

### Coordinate Systems

- **RTB**: Standard robotics conventions
- **Genesis**: Right-handed coordinate system
- **ManiPylator**: Consistent interface between both

## Common Tasks

### Loading Different Robots
```python
# Use built-in templates
with render_robot_from_template("robots/empiric") as urdf:
    robot = VisualRobot(urdf)

# Load your own URDF
from pathlib import Path
robot = VisualRobot(Path("my_robot.urdf"))
```

### Getting Robot Information
```python
# View robot structure
print(robot.model)

# Get current joint angles
print(robot.model.q)

# Get end effector position
translation, rotation = robot.get_transformation_matrix()
```

### Running Simulations
```python
# Visual simulation (with 3D viewer)
robot = SimulatedRobot(urdf_path)

# Headless simulation (faster, no GUI)
robot = HeadlessSimulatedRobot(urdf_path)

# Step to a new pose
translation, rotation = robot.step_to_pose([0, 0, 0, 0, 0, 0])
```

**Ready to start?** Open `00-start-here.ipynb` and run your first robot simulation!