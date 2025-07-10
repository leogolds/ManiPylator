# ManiPylator
<img src="logo.webp" width="150" height="150" align="right">

3D printed 6DOF robotic manipulator powered by your favorite snake based programming language

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

## Features

- **Hardware Control**: Direct integration with Klipper for precise stepper motor control
- **Simulation**: Genesis-powered 3D visualization and physics simulation
- **Motion Planning**: Advanced kinematics and trajectory planning algorithms
- **Cloud Ready**: Containerized with CI/CD via Dagger
- **Learning Focused**: Comprehensive notebooks bridging theory and practice

## Documentation

For detailed examples and tutorials, check out the blog posts:
- [Part 1: Where to start?](https://hackaday.io/project/197770-manipylator/log/232565-manipilator-part-1-where-to-start)
- [Part 2: Simulation & Motion Planning](https://hackaday.io/project/197770-manipylator/log/240946-manipylator-part-2-simulation-motion-planning)