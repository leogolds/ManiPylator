# Frequently Asked Questions (FAQ)

## Installation & Setup

### Q: What are the system requirements?
A: ManiPylator requires:
- Python 3.10 or higher
- CUDA-compatible GPU recommended (Genesis can fall back to CPU but runs slower)
- Linux (recommended) or Windows with WSL
- At least 4GB RAM, 8GB recommended

### Q: I'm getting CUDA errors. What should I do?
A:
The lab should run in a containerized environment and can be provided a GPU (see [lab compose file](../compose.yaml)). Either way, Genesis gracefully downgrades to running on CPU if one is not provided/available. A useful tool to check on the GPU is `nvidia-smi`.
1. Ensure you have CUDA drivers installed
2. Check that your GPU supports CUDA
3. Try running with CPU backend: `gs.init(backend=gs.cpu)`
4. Verify PyTorch CUDA installation: `python -c "import torch; print(torch.cuda.is_available())"`

### Q: I don't have a GPU. Can I still use ManiPylator?
A: Yes. Genesis gracefully falls back to CPU mode (`gs.init(backend=gs.cpu)`),
though simulation will be slower. For kinematics-only work you can also use
the `minimal` Docker profile and RTB-only notebooks -- symbolic math,
Jacobians, trajectory planning, and MQTT development all work without a GPU.
See [Lab Environment Configurations](RESOURCES.md#lab-environment-configurations)
for what's possible at each level.

### Q: Genesis takes a very long time to start. Is that normal?
A: Yes, on first run. Genesis startup involves two expensive steps:

1. **Geometry preprocessing** -- mesh processing for each URDF link. Cached in
   the `genesis-cache` Docker volume after the first run.
2. **Taichi kernel compilation** -- GPU kernel compilation. The
   "Compiling simulation kernels..." log message always appears, even when
   loading from cache, but cached loading is much faster.

Approximate times on a low-end GPU (MX250, 2GB VRAM):

| Run | Time |
|---|---|
| Cold (no cache) | ~8 min |
| Geometry cached only | ~4 min |
| Fully cached | ~1.5 min |

Faster GPUs will see shorter times. Subsequent runs reuse the cache
automatically as long as the `genesis-cache` Docker volume persists.

### Q: The Genesis kernel cache doesn't seem to persist between runs
A: Taichi only writes the kernel cache on clean exit. Always stop Genesis
processes with Ctrl-C or `sys.exit()`, never SIGKILL or a hard container stop.
Look for the log message "Exiting Genesis and caching compiled kernels..." to
confirm the cache was written. The `run_mq_visualizer.py` handles this
automatically via its SIGINT signal handler.

## Usage

### Q: How do I load a robot?
A: Use the template rendering system:
```python
from manipylator import HeadlessSimulatedRobotDevice
from manipylator.utils import render_robot_from_template

with render_robot_from_template("robots/empiric") as robot_urdf:
    robot = HeadlessSimulatedRobotDevice(robot_urdf)
```

Alternatively, both [roboticstoolbox-python](https://petercorke.github.io/robotics-toolbox-python/intro.html#robot-models) and [Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html#load-objects-into-the-scene) provide their own methods of loading standard robotics formats. YMMV.

### Q: What's the difference between SimulatedRobotDevice and HeadlessSimulatedRobotDevice?
A:
- **SimulatedRobotDevice**: Includes 3D visualization with Genesis viewer (requires X11 forwarding)
- **HeadlessSimulatedRobotDevice**: Runs simulation without GUI (faster, good for servers and CI)

### Q: The Genesis viewer window won't open
A: You likely forgot to run `xhost +local:root` before starting the container.
This allows the container to access your X11 display. If X11 isn't available at
all, use `HeadlessSimulatedRobotDevice` instead.

### Q: My robot moves unexpectedly or joint angles seem wrong
A: The entire stack uses **radians** -- both simulation and Klipper gcode.
The Klipper stepper configuration sets `rotation_distance` to 2*pi so that
gcode values are interpreted as radians directly. If you see unexpected
behavior, double-check that you haven't accidentally passed degree values.

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

### Address already in use when starting Docker services

Docker publishes host ports for the stack (for example **1883** for Mosquitto `mq`, **6379** for Redis `tq`, **8888** for Jupyter `lab`). If something else on the machine is already bound to that port, Compose fails with an error like:

`failed to bind host port 0.0.0.0:1883/tcp: address already in use`

**Find what is using the port (Linux):**

```bash
sudo ss -tlnp | grep ':1883'
# or, if installed:
sudo lsof -i :1883
```

The output includes a **PID** (process id). Typical causes: a system `mosquitto` or `redis-server` service, another Docker container publishing the same port, or a leftover process after an unclean exit.

**Free the port:**

- If you recognize the process and it is yours: `kill <pid>`, or `kill -9 <pid>` if it does not exit on SIGTERM.
- If it is another container: `docker ps` (and `docker ps -a`) to find it, then `docker stop <container>`.
- As a last resort on a dev machine only: `sudo fuser -k 1883/tcp` kills whatever holds that port (replace `1883` with the conflicting port).

If a different Compose project or stack is using the same published ports, stop that stack or change the host port mapping in the relevant `compose.yaml`.

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
A: Yes! ManiPylator supports MQTT communication for real robot control. Use
`PhysicalRobotDevice` to send gcode commands to Klipper. See the
`30-controlling-manny.ipynb` notebook and the [API Reference](API_REFERENCE.md).

### Q: How do I record videos of simulations?
A: Use Genesis camera recording:
```python
camera = robot.simulator.camera
camera.start_recording()
# ... run simulation ...
camera.stop_recording(save_to_filename='video.mp4', fps=20)
```

### Q: What coordinate systems are used?
A:
- **RTB**: Uses standard robotics conventions
- **Genesis**: Uses right-handed coordinate system
- **ManiPylator**: Provides consistent interface between both
- **Joint angles**: Radians throughout (simulation and Klipper gcode)

## Performance

### Q: How can I improve simulation performance?
A:
1. Use `HeadlessSimulatedRobotDevice` for faster simulation
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
4. Check the [API Reference](API_REFERENCE.md)
5. Open an issue on GitHub
6. Check the [Getting Started](GETTING_STARTED.md) guide

### Q: How do I report a bug?
A:
1. Check if it's already reported in GitHub issues
2. Create a new issue with:
   - Clear description of the problem
   - Steps to reproduce
   - System information
   - Error messages/logs

---

*For more detailed information, see the [Architecture](ARCHITECTURE.md), [API Reference](API_REFERENCE.md), and [Getting Started](GETTING_STARTED.md) documentation.*
