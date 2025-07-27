# Frequently Asked Questions (FAQ)

## Installation & Setup

### Q: What are the system requirements?
A: ManiPylator requires:
- Python 3.10 or higher
- CUDA-compatible GPU (for Genesis simulation)
- Linux (recommended) or Windows with WSL
- At least 4GB RAM, 8GB recommended

### Q: I'm getting CUDA errors. What should I do?
A: 
The lab should run in a containerized environment and can be provided a GPU (see [lab compose file](compose.yaml)). Either way, Genesis gracefully downgrades to running on CPU if one is not provided/available. A useful tool to check on the GPU is `nvidia-smi`.
1. Ensure you have CUDA drivers installed
2. Check that your GPU supports CUDA
3. Try running with CPU backend: `gs.init(backend=gs.cpu)`
4. Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

## Usage

### Q: How do I load a robot?
A: Use the template rendering system:
```python
from manipylator import VisualRobot
from manipylator.utils import render_robot_from_template

with render_robot_from_template("robots/empiric") as robot_urdf:
    robot = VisualRobot(robot_urdf)
```

Alternatively, both [roboticstoolbox-python](https://petercorke.github.io/robotics-toolbox-python/intro.html#robot-models) and [Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html#load-objects-into-the-scene) provide their own methods of loading stadard robotics formats. YMMV.

### Q: What's the difference between VisualRobot and HeadlessVisualRobot?
A:
- **VisualRobot**: Includes 3D visualization with Genesis viewer
- **HeadlessVisualRobot**: Runs simulation without GUI (faster, good for servers)


## Troubleshooting

### Q: URDF loading errors
A:
1. Ensure all mesh files are present in the assets directory
2. Check URDF file syntax
3. Use the template system for proper path resolution
4. Verify mesh file formats (STL recommended)

### Q: Camera rendering issues
A:
1. For color issues, use `hv.RGB()` instead of `hv.Image()`
2. Ensure proper aspect ratio: `hv.RGB(frame).opts(aspect=frame.shape[1] / frame.shape[0])`
3. Set pixel bounds: `hv.RGB(frame, bounds=(0, 0, width, height))`

### Q: MQTT connection problems
A:
1. Check if MQTT broker is running
2. Verify host address and port
3. Check network connectivity
4. Ensure proper topic subscriptions

## Development

### Q: How do I add a new robot?
A:
1. Create a new directory in `robots/`
2. Add your URDF template as `robot.urdf.j2`
3. Create an `assets/` subdirectory with STL files
4. Use `render_robot_from_template()` to load it

### Q: How do I contribute to the project?
A:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request


## Project-Specific

### Q: What is the difference between RTB and Genesis?
A:
- **RTB (Robotics Toolbox)**: Analytical/symbolic robotics calculations
- **Genesis**: Physics-based simulation with 3D rendering
- **ManiPylator**: Provides unified interface to both

### Q: Can I use ManiPylator with real hardware?
A: Yes! ManiPylator supports MQTT communication for real robot control. See the MQTT examples and physical robot notebooks.

### Q: How do I record videos of simulations?
A: Use Genesis camera recording:
```python
camera = robot.visualizer.camera
camera.start_recording()
# ... run simulation ...
camera.stop_recording(save_to_filename='video.mp4', fps=20)
```

### Q: What coordinate systems are used?
A:
- **RTB**: Uses standard robotics conventions
- **Genesis**: Uses right-handed coordinate system
- **ManiPylator**: Provides consistent interface between both

## Performance

### Q: How can I improve simulation performance?
A:
1. Use `HeadlessVisualRobot` for faster simulation
2. Reduce simulation substeps
3. Use lower resolution cameras
4. Enable Genesis performance mode: `gs.init(performance_mode=True)`

### Q: Memory usage is high
A:
1. Close unused robot instances
2. Use headless mode when visualization isn't needed
3. Reduce camera resolution
4. Clear debug objects: `scene.clear_debug_objects()`

## Support

### Q: Where can I get help?
A:
1. Check this FAQ first
2. Look at the example notebooks
3. Check the [Architecture](ARCHITECTURE.md) documentation
4. Open an issue on GitHub
5. Check the [Getting Started](GETTING_STARTED.md) guide

### Q: How do I report a bug?
A:
1. Check if it's already reported in GitHub issues
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - System information
   - Error messages/logs

---

*For more detailed information, see the [Architecture](ARCHITECTURE.md) and [Getting Started](GETTING_STARTED.md) documentation.* 