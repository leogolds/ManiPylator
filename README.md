# ManiPylator
<img src="docs/logo.webp" width="150" height="150" align="right">

3D printed 6DOF robotic manipulator powered by your favorite snake based programming language

## What is this?

**ManiPylator** is an open-source project developing an accessible, high-quality & reproducible stack for robotics experimentation. Our goal is to lower the barrier to entry for both hobbyists and researchers by providing a modern develoment environment based around containers. Furthermore, **ManiPylator** aims to be a "batteries included" robotics platform by bringing together robust educational content backed by the [Genesis](https://genesis-world.readthedocs.io/en/latest/#) physics simulation engine & the Python scientific stack. 

This project, in part, came about as an attempt to build a 3D printed, 6DOF robotic manipulator, programmable in Python. While alternatives exist, finding a solution that can take you from a 3D model to a moving robot proved challanging. This project aims to do just that, include everything you need to go from ideation to simulation within a modern software envirinoment. By developing a stable platform for embodied robotics in an open-source context we hope to spark experimentation & innovation.

---

## Where should I go?

- **I know nothing of robotics but excited to learn**: many paths lead here, but which is best for you? See the RESOURCES.md file for some high quality introductory robotics resources. For those seeking a narrative introduction, check out [Part 1: Where to start?](https://hackaday.io/project/197770-manipylator/log/232565-manipilator-part-1-where-to-start) and [Part 2: Simulation & Motion Planning](https://hackaday.io/project/197770-manipylator/log/240946-manipylator-part-2-simulation-motion-planning)
- **I'm a student/researcher/teacher**: experimentation is a cornerstone of learning. This project tries to provide access to state of the art robotics software in a user-friendly, no-nonesense package. See the the [Getting Started guide](docs/GETTING_STARTED.md), or take a look at the included notebooks to get a feel for what's available. For example, the notebooks starting with 1X demonstrate symbolic math using [SymPy](https://docs.sympy.org/) and how to manipulate [SE2 & SE3](https://bdaiinstitute.github.io/spatialmath-python/intro.html#spatial-math-classes) objects. These skills are then applied to derive introductory robotics results.
- **I'm a robotocist and would like to help with developemnt**: check out the [CONTRIBUTING.md](CONTRIBUTING.md) file or open an issue. To learn how to build the project locally check out out [Local Development instructions](docs/LOCAL_DEVELOPMENT.md)
- **I'm just curious. How does this work?**: Please check out our [Architecture](docs/ARCHITECTURE.md) outline to understand how things are put together.

## Basic Setup
``` bash
git clone https://github.com/leogolds/ManiPylator.git
cd ManiPylator

xhost +local:root # Allow the container to access the display

docker compose up lab -d
```
Now go take a look at our [Start Here notebook](http://localhost:8888/lab/tree/00-start-here.ipynb) to try out a few basics