
## Local Development

This project uses [Dagger](https://dagger.io/) for CI/CD and development workflows.


### Prerequisites
- [Dagger CLI](https://docs.dagger.io/quickstart/cli)
```bash
curl -fsSL https://dl.dagger.io/dagger/install.sh | DAGGER_VERSION=0.18.12 BIN_DIR=$HOME/.local/bin sh
```
- Docker

### Deploying daggger-engine

For optimal local development experience, we provide a custom Dagger engine configuration with optimized caching and garbage collection settings. See [`containers/dagger-engine/README.md`](containers/dagger-engine/README.md) for detailed instructions on:

- Setting up custom engine configuration for faster builds
- Managing cache dependencies and disk usage
- Configuring garbage collection for your development environment
- Troubleshooting common engine issues

The configuration includes optimized settings for:
- **Cache Management**: 35GB max cache with intelligent garbage collection
- **Disk Space**: Reserved space protection and automatic cleanup
- **Build Performance**: Optimized builds with extensive caching

#### Quick Engine Setup
```bash
# Reload Dagger engine with custom configuration
cd containers/dagger-engine
./reload-dagger-engine.sh

# Set environment variable to use the custom engine
export _EXPERIMENTAL_DAGGER_RUNNER_HOST=docker-container://dagger-engine-custom
```

This script automatically stops any existing Dagger engines, applies your custom configuration, and starts a new engine with custom caching settings for faster builds. The environment variable tells the Dagger CLI to connect to your custom engine instead of the default one.


### Available Commands

#### Build Containers
```bash
# Build the lab environment container
dagger call build-lab --source=.

# Build the controller container  
dagger call build-controller --source=.

# Publish lab container to registry
dagger call publish-lab --source=. --tag=latest
```

#### Camera Service Container
```bash
# Build for AMD64 (native) and export as tarball
dagger call camera-container --source=. \
  export --path=camera-service-amd.tar

# Build for ARM64 (Raspberry Pi) and export as tarball
dagger call camera-container --source=. --platform=linux/arm64 \
  export --path=camera-service-arm.tar

# Publish AMD64 variant to Docker Hub
dagger call publish-camera-container --source=. \
  --tag=latest \
  --registry-username=leogold \
  --registry-password=env:DOCKER_TOKEN

# Publish ARM64 variant to Docker Hub
dagger call publish-camera-container --source=. \
  --platform=linux/arm64 \
  --tag=latest \
  --registry-username=leogold \
  --registry-password=env:DOCKER_TOKEN
```

Images are tagged as `leogold/manipylator:camera-service-{amd|arm}-{tag}`.

To run on a machine with USB cameras (see `containers/camera/compose.yaml`):
```bash
cd containers/camera

# AMD64 (default)
docker compose up -d

# ARM64 (Raspberry Pi)
ARCH=arm docker compose up -d
```

#### Profiling the Camera Service

Scripts in `manual_tests/` help verify and profile a running camera and pipeline.
They expect the camera service, MQTT broker, and (for pipeline tests) Redis + Huey
workers to be running.

```bash
# End-to-end pipeline profiler -- camera health, REST latency, MJPEG FPS,
# frame staleness, and analysis-pipeline stats in a single report
python manual_tests/test_pipeline_profile.py
python manual_tests/test_pipeline_profile.py \
  --rest-url http://192.168.1.50:8001 \
  --stream-url http://192.168.1.50:8000/video \
  --duration 30

# Frame lifecycle profiler -- frame age and MediaPipe processing time
python manual_tests/test_frame_profiling.py

# REST vs MJPEG latency comparison and MQTT hand-guard event timing
python manual_tests/test_latency.py --duration 60

# Huey task round-trip latency (needs Huey workers running)
python manual_tests/test_huey_performance.py

# MediaPipe hand detection on fixed/live images (no pipeline required)
python manual_tests/test_hand_detection.py --show
python manual_tests/test_hand_detection.py --live-monitor

# MQTT stream discovery -- verify cameras publish StreamInfoV1
python manual_tests/test_discovery.py
python manual_tests/test_discovery.py --interactive
```

#### Development Stack
```bash
# Spin up full stack in virtual mode
dagger call controller up --source=. --virtual-mode=true

# Run system tests
dagger call controller test --source=.

# Run a simulation notebook
dagger call controller simulate --source=. --notebook=generate-trajectory-example.ipynb
```

## Traditional Docker Compose

For local development, you can still use Docker Compose:

```bash
# Spin up jupyter lab environment
docker compose up -d

# Allow X11 forwarding for Genesis window
xhost +local:root
```

Go to [http://localhost:8888/lab/workspaces/auto-0/tree/generate-trajectory-example.ipynb](http://localhost:8888/lab/workspaces/auto-0/tree/generate-trajectory-example.ipynb) for a short intro. The notebook demonstrats loading a URDF, forward/inverse kinematics, and saving control signals as a csv.

## Project Structure

- `manipylator/` - Python library for robotics exploration
- `controller/` - Klipper stack + MQTT integration + 6DOF movement macros  
- `robots/` - URDF models and robot configurations
- `containers/` - Docker containers for lab environment
- `dagger/` - Dagger CI/CD module for builds and deployment
- `*.ipynb` - Jupyter notebooks with examples and tutorials
