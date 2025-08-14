# ManiPylator
<img src="docs/logo.webp" width="150" height="150" align="right">

3D printed 6DOF robotic manipulator powered by your favorite snake based programming language

## What is this?

**ManiPylator** is an open-source project developing an accessible, high-quality & reproducible stack for robotics experimentation. Our goal is to lower the barrier to entry for both hobbyists and researchers by providing a modern robotics develoment environment based around containers. Furthermore, **ManiPylator** aims to be a "batteries included" robotics platform by bringing together robust educational content backed by the [Genesis](https://genesis-world.readthedocs.io/en/latest/#) physics simulation engine, [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python) & the Python scientific stack. 

This project, in part, came about as an attempt to build a 3D printed, 6DOF robotic manipulator, programmable in Python. While alternatives exist, finding a solution that can take you from a 3D model to a moving robot proved challanging. This project aims to do just that, include everything you need to go from ideation to simulation within a modern software envirinoment. By developing a stable platform for embodied robotics in an open-source context we hope to spark experimentation & innovation.

---

## Where should I go?

- **I know nothing of robotics but excited to learn**: many paths lead here, but which is best for you? See the RESOURCES.md file for some high quality introductory robotics resources. For those seeking a narrative introduction, check out [Part 1: Where to start?](https://hackaday.io/project/197770-manipylator/log/232565-manipilator-part-1-where-to-start) and [Part 2: Simulation & Motion Planning](https://hackaday.io/project/197770-manipylator/log/240946-manipylator-part-2-simulation-motion-planning)
- **I'm a student/researcher/teacher**: experimentation is a cornerstone of learning. This project aims to provide access to state of the art robotics software in a user-friendly, no-nonesense package. See the the [Getting Started guide](docs/GETTING_STARTED.md), or take a look at the included notebooks to get a feel for what's available. For example, the notebooks starting with 1X demonstrate symbolic math using [SymPy](https://docs.sympy.org/) and how to manipulate [SE2 & SE3](https://bdaiinstitute.github.io/spatialmath-python/intro.html#spatial-math-classes) objects. These skills are then applied to derive introductory robotics results. Check out [`10-symbolic-manipulation.ipynb`](https://nbviewer.org/github/leogolds/ManiPylator/blob/main/10-symbolic-manipulation.ipynb) and [`11-differntial-kinematics.ipynb`](https://nbviewer.org/github/leogolds/ManiPylator/blob/main/11-differntial-kinematics.ipynb) for examples.
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

## Included Tutorial Collections

This repository includes two excellent tutorial collections that provide comprehensive coverage of robotics fundamentals:

**Textbook Reference**: These tutorials complement the comprehensive robotics textbook [*Robotics, Vision and Control*](https://petercorke.com/wordpress/books/robotics-vision-control/) by Peter Corke. For more details, visit [Peter Corke's website](https://petercorke.com/).

### 1. Differential Kinematics Tutorial (DKT)
**Source**: [jhavl/dkt](https://github.com/jhavl/dkt) by Jesse Haviland and Peter Corke

**Part 1**: Kinematics, Velocity, and Applications
- [Manipulator Kinematics](http://localhost:8888/lab/tree/external/dkt/Part%201/1%20Manipulator%20Kinematics.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%201/1%20Manipulator%20Kinematics.ipynb)
- [The Manipulator Jacobian](http://localhost:8888/lab/tree/external/dkt/Part%201/2%20The%20Manipulator%20Jacobian.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%201/2%20The%20Manipulator%20Jacobian.ipynb)
- [Resolved-Rate Motion Control](http://localhost:8888/lab/tree/external/dkt/Part%201/3%20Resolved-Rate%20Motion%20Control.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%201/3%20Resolved-Rate%20Motion%20Control.ipynb)
- [Numerical Inverse Kinematics](http://localhost:8888/lab/tree/external/dkt/Part%201/4%20Numerical%20Inverse%20Kinematics.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%201/4%20Numerical%20Inverse%20Kinematics.ipynb)
- [Manipulator Performance Measures](http://localhost:8888/lab/tree/external/dkt/Part%201/5%20Manipulator%20Performance%20Measures.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%201/5%20Manipulator%20Performance%20Measures.ipynb)

**Part 2**: Acceleration and Advanced Applications
- [The Manipulator Hessian](http://localhost:8888/lab/tree/external/dkt/Part%202/1%20The%20Manipulator%20Hessian.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%202/1%20The%20Manipulator%20Hessian.ipynb)
- [Higher Order Derivatives](http://localhost:8888/lab/tree/external/dkt/Part%202/2%20Higher%20Order%20Derivatives.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%202/2%20Higher%20Order%20Derivatives.ipynb)
- [Analytic Forms](http://localhost:8888/lab/tree/external/dkt/Part%202/3%20Analytic%20Forms.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%202/3%20Analytic%20Forms.ipynb)
- [Null-Space Projection for Motion Control](http://localhost:8888/lab/tree/external/dkt/Part%202/4%20Null-Space%20Projection%20for%20Motion%20Control.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%202/4%20Null-Space%20Projection%20for%20Motion%20Control.ipynb)
- [Quadratic Programming for Motion Control](http://localhost:8888/lab/tree/external/dkt/Part%202/5%20Quadratic%20Programming%20for%20Motion%20Control.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%202/5%20Quadratic%20Programming%20for%20Motion%20Control.ipynb)
- [Advanced Numerical Inverse Kinematics](http://localhost:8888/lab/tree/external/dkt/Part%202/6%20Advanced%20Numerical%20Inverse%20Kinematics.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%202/6%20Advanced%20Numerical%20Inverse%20Kinematics.ipynb)
- [Quadratic-Rate Motion Control](http://localhost:8888/lab/tree/external/dkt/Part%202/7%20Quadratic-Rate%20Motion%20Control.ipynb) | [Source](https://github.com/jhavl/dkt/blob/main/Part%202/7%20Quadratic-Rate%20Motion%20Control.ipynb)

### 2. Spatial Mathematics Tutorial
**Source**: [jhavl/spatialmathematics](https://github.com/jhavl/spatialmathematics) by Jesse Haviland and Peter Corke

**Core Concepts**:
- [Linear Transformations](http://localhost:8888/lab/tree/external/spatialmathematics/0%20Linear%20Transformations.ipynb) | [Source](https://github.com/jhavl/spatialmathematics/blob/main/0%20Linear%20Transformations.ipynb)
- [Spatial Mathematics Fundamentals](http://localhost:8888/lab/tree/external/spatialmathematics/1%20Spatial%20Mathematics.ipynb) | [Source](https://github.com/jhavl/spatialmathematics/blob/main/1%20Spatial%20Mathematics.ipynb)
- [Deep Dive into Orientation (quaternions, Euler angles, rotation matrices)](http://localhost:8888/lab/tree/external/spatialmathematics/2%20Deep%20Dive%20into%20Orientation.ipynb) | [Source](https://github.com/jhavl/spatialmathematics/blob/main/2%20Deep%20Dive%20into%20Orientation.ipynb)
- [Twists and Trajectories](http://localhost:8888/lab/tree/external/spatialmathematics/3%20Twists%20and%20Trajectories.ipynb) | [Source](https://github.com/jhavl/spatialmathematics/blob/main/3%20Twists%20and%20Trajectories.ipynb)
- [Differential Spatial Mathematics](http://localhost:8888/lab/tree/external/spatialmathematics/4%20Differential%20Spatial%20Mathematics.ipynb) | [Source](https://github.com/jhavl/spatialmathematics/blob/main/4%20Differential%20Spatial%20Mathematics.ipynb)
- [Graph Theory and Planning Algorithms](http://localhost:8888/lab/tree/external/spatialmathematics/5%20Graph%20Theory%20and%20Planning%20Algorithms.ipynb) | [Source](https://github.com/jhavl/spatialmathematics/blob/main/5%20Graph%20Theory%20and%20Planning%20Algorithms.ipynb)

**Location**: `external/spatialmathematics/`

## Learning Path

For beginners, we recommend starting with the **Spatial Mathematics Tutorial** to build foundational knowledge, then progressing to the **Differential Kinematics Tutorial** for advanced manipulator concepts.

Both tutorial collections are powered by the [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python) and provide hands-on examples perfect for learning robotics concepts.