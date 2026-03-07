# Learning Resources

This page collects external references, learning paths, hands-on experiments,
and teaching material for using ManiPylator alongside a robotics curriculum.

---

## Learning Paths

Recommended notebook progressions depending on your level. Chapter references
are to Corke, [*Robotics, Vision and Control*](https://link.springer.com/book/10.1007/978-3-031-06469-2), 3rd ed. (Python).

### Beginner (new to robotics -- covers Ch 2, 3, 7.1)

`00-start-here` &rarr; `10-symbolic-manipulation` &rarr;
`external/spatialmathematics/0..2` (Ch 2) &rarr; `1x-forward-kinematics*`
(Ch 7.1) &rarr; `extra-notebooks/generate-trajectory-example` (Ch 3) &rarr;
`20-simulation`

### Intermediate (comfortable with FK, learning IK and control -- covers Ch 7.2-7.4, 8)

`1x-inverse-kinematics*` (Ch 7.2) &rarr; `external/dkt/Part 1` (Ch 8) &rarr;
`21-headless-simulation` &rarr;
`extra-notebooks/analyzing-trajectories-using-sympy-numpy-hvplot` (Ch 7.3)

### Advanced (Jacobians, dynamics, motion control, real hardware -- covers Ch 8-9, 15)

`external/dkt/Part 2` (Ch 8 advanced, App E) &rarr; `30-controlling-manny`
(Ch 9) &rarr; `run_pipeline.py` (Ch 12, 15) &rarr; custom `MQClient` device
development

### Classroom (no GPU required -- covers Ch 2-3, 7-8)

Use the `minimal` Docker profile (`docker compose --profile minimal up -d`).
All RTB-based notebooks work without Genesis: `10-*`, `1x-*`,
`external/spatialmathematics/`, `external/dkt/`. Symbolic math, FK, IK,
Jacobians, and motion control can all be taught without GPU hardware or a
physical robot.

---

## Concepts and Capabilities

Quick-reference table mapping robotics concepts to project notebooks and API.

| Robotics Concept | Notebooks | API / Code |
|---|---|---|
| Spatial mathematics (SE2, SE3, twists) | `external/spatialmathematics/` | `spatialmath` library |
| Forward kinematics (DH parameters) | `1x-forward-kinematics*` | `robot.model.fkine()` |
| Inverse kinematics (analytical + numerical) | `1x-inverse-kinematics*` | `robot.model.ikine_LM()` |
| Jacobians and velocity kinematics | `external/dkt/Part 1/2..3` | `robot.model.jacob0()` |
| Hessians and higher-order derivatives | `external/dkt/Part 2/1..2` | `robot.model.hessian0()` |
| Trajectory planning | `extra-notebooks/generate-trajectory-example` | `manipylator.utils` parametric curves, `rtb.jtraj()` |
| Motion control (resolved-rate, QP, null-space) | `external/dkt/Part 1/3`, `Part 2/4..7` | RTB controllers |
| Symbolic manipulation | `10-symbolic-manipulation` | SymPy |
| GPU physics simulation | `20-simulation`, `21-headless-simulation` | `SimulatedRobotDevice`, `HeadlessSimulatedRobotDevice` |
| Physical robot control | `30-controlling-manny` | `PhysicalRobotDevice`, Klipper gcode |
| Computer vision / safety | `run_pipeline.py` | `StreamingCamera`, `HandDetector`, `SafetyListener` |
| MQTT device communication | `extra-notebooks/comms-control` | `MQClient`, `MQTTConnection` |

---

## Textbook Chapter Mapping (Corke, RVC 3rd ed. Python)

The `external/` tutorial collections and ManiPylator's own notebooks are
companions to [*Robotics, Vision and Control*](https://link.springer.com/book/10.1007/978-3-031-06469-2) (3rd ed., Springer 2023) by
Peter Corke. Coverage level: **Full** = dedicated notebooks + API,
**Partial** = API or exercises available, **Context** = relevant background
but no direct tooling.

### Part I: Foundations

| Chapter | Topic | Coverage | ManiPylator Resources |
|---|---|---|---|
| Ch 2 | Coordinate frames, SE2/SE3, rotation matrices, Euler angles, quaternions, twists | Full | `external/spatialmathematics/0..5`, `10-symbolic-manipulation.ipynb`, `spatialmath` library |
| Ch 2.3 | 3D rotations, quaternions, angle-axis | Full | `external/spatialmathematics/2 Deep Dive into Orientation`, `manipylator.utils.quaternion_to_rotation_matrix()` |
| Ch 2.4 | Exponential mapping, Lie groups | Full | `external/spatialmathematics/3 Twists and Trajectories`, `external/spatialmathematics/4 Differential Spatial Mathematics` |
| Ch 3 | Trajectories, time-varying pose, smooth motion | Full | `extra-notebooks/generate-trajectory-example`, `rtb.jtraj()`, `rtb.ctraj()`, `manipylator.utils` parametric curves |
| Ch 3.3 | Quintic polynomials, multi-axis, orientation interpolation | Full | `extra-notebooks/analyzing-trajectories-using-sympy-numpy-hvplot`, `rtb.jtraj()` |

### Part II: Mobile Robotics (Chapters 4-6)

Not directly applicable to ManiPylator (which is a manipulator arm). However,
the `external/spatialmathematics/5 Graph Theory and Planning Algorithms`
notebook covers graph-based planning concepts from Ch 5.

### Part III: Robot Manipulators

| Chapter | Topic | Coverage | ManiPylator Resources |
|---|---|---|---|
| Ch 7.1 | Pose graphs, DH parameters, ETS, URDF | Full | `1x-forward-kinematics*.ipynb`, `10-symbolic-manipulation.ipynb`, `robot.model.fkine()`, URDF templates in `robots/` |
| Ch 7.1.4 | Unified Robot Description Format | Full | `robots/empiric/robot.urdf.j2`, `robots/vanilla/robot.urdf.j2`, `render_robot_from_template()` |
| Ch 7.1.5 | DH convention, link transforms | Full | `1x-forward-kinematics*.ipynb`, `10-symbolic-manipulation.ipynb` (SymPy DH derivation) |
| Ch 7.2 | Analytical (2D/3D), numerical IK, redundancy | Full | `1x-inverse-kinematics*.ipynb`, `robot.model.ikine_LM()`, `external/dkt/Part 1/4 Numerical Inverse Kinematics` |
| Ch 7.3 | Joint-space, Cartesian, singularity traversal | Full | `extra-notebooks/generate-trajectory-example`, `extra-notebooks/control-move-in-heart-shape`, `rtb.jtraj()` |
| Ch 7.4 | Cartesian path following | Full | `extra-notebooks/control-move-in-heart-shape` (parametric heart trajectory on the Empiric arm) |
| Ch 8.1 | World-frame, end-effector-frame, analytical Jacobian | Full | `external/dkt/Part 1/2 The Manipulator Jacobian`, `robot.model.jacob0()`, `robot.model.jacobe()` |
| Ch 8.2 | Velocity-based Cartesian control | Full | `external/dkt/Part 1/3 Resolved-Rate Motion Control` |
| Ch 8.3 | Singularities, velocity ellipsoid, manipulability index | Full | `external/dkt/Part 1/5 Manipulator Performance Measures` |
| Ch 8.4 | Wrench mapping, force ellipsoids | Partial | `robot.model.jacob0()` (transpose for force mapping), no dedicated notebook |
| Ch 8.5 | Jacobian-based IK, Levenberg-Marquardt | Full | `external/dkt/Part 1/4 Numerical Inverse Kinematics`, `external/dkt/Part 2/6 Advanced Numerical Inverse Kinematics` |
| Ch 9 | Equations of motion, gravity, inertia, Coriolis | Partial | `robot.model.gravload()`, `robot.model.inertia()`, `robot.model.coriolis()` -- API available but no dedicated notebook |
| Ch 9.4-9.5 | Feedforward, computed-torque, operational space | Partial | `external/dkt/Part 2/4 Null-Space Projection`, `external/dkt/Part 2/5 Quadratic Programming` |

### Part IV: Computer Vision (Chapters 10-14)

| Chapter | Topic | Coverage | ManiPylator Resources |
|---|---|---|---|
| Ch 11 | Image acquisition, filtering, morphology | Partial | `StreamingCamera` (MJPEG + REST), OpenCV available in lab container |
| Ch 12.1.4 | Deep learning-based detection | Context | `HandDetector` uses MediaPipe (related approach); extensible via the pipeline |
| Ch 13 | Camera models, calibration, projection | Partial | `22-camera-controls.ipynb` (Genesis camera), Genesis camera intrinsics/extrinsics |

### Part V: Vision-Based Control (Chapters 15-16)

| Chapter | Topic | Coverage | ManiPylator Resources |
|---|---|---|---|
| Ch 15 | PBVS, IBVS, image Jacobian | Context | `StreamingCamera` + `PhysicalRobotDevice` provide the infrastructure for visual servoing experiments; no pre-built IBVS loop |
| Ch 16 | Arm-type robot servoing | Context | Could be built using `MQClient` + `StreamingCamera` + `SimulatedRobotDevice` |

### Appendices

| Appendix | Topic | Coverage | ManiPylator Resources |
|---|---|---|---|
| App B | Vectors, matrices, eigenvalues, SVD | Context | NumPy in lab container |
| App D | SO(3), SE(3), exponential map | Full | `external/spatialmathematics/3..4`, `spatialmath` library |
| App E | Derivatives, Taylor expansion | Full | `external/dkt/Part 2/1 The Manipulator Hessian`, `external/dkt/Part 2/2 Higher Order Derivatives` |

---

## Video Courses and References

- [bioMechatronics Lab lecture series](https://www.youtube.com/playlist?list=PLY6RHB0yqJVasji1rwZAGYirD8zW1ipj-) -- comprehensive course from basics to surgical robotics. Key lectures: [rigid transformations](https://www.youtube.com/watch?v=1-_HhBlQRp8&list=PLY6RHB0yqJVasji1rwZAGYirD8zW1ipj-&index=2) (Ch 2), [forward kinematics](https://www.youtube.com/watch?v=6sd9fiinq5U&list=PLY6RHB0yqJVasji1rwZAGYirD8zW1ipj-&index=5) (Ch 7), [inverse kinematics](https://www.youtube.com/watch?v=RzaeS5LLhxA&list=PLY6RHB0yqJVasji1rwZAGYirD8zW1ipj-&index=7) (Ch 7.2)
- [Robotics Explained](https://robotics-explained.com/) -- interactive 2D demos for [forward](https://robotics-explained.com/forwardkinematics) and [inverse](https://robotics-explained.com/inversekinematics) kinematics (Ch 7)
- [*Robotics, Vision and Control*](https://link.springer.com/book/10.1007/978-3-031-06469-2) by Peter Corke ([author's website](https://petercorke.com/rvc3p/home/)) -- the textbook behind the `external/` tutorial collections

### Quaternion Resources (Ch 2.3)

- [Practical quaternion intro (short)](https://www.youtube.com/watch?v=bKd2lPjl92c)
- [Freya Holmer's quaternion explainer (long)](https://www.youtube.com/watch?v=PMvIWws8WEo)
- [SymPy quaternion docs](https://docs.sympy.org/latest/modules/algebras.html)
- [Rotation matrix to quaternion derivation (PDF)](https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf)
- [How Maxwell's Equations (and Quaternions) Led to Vector Analysis](https://www.youtube.com/watch?v=M12CJIuX8D4) -- the story of Tait, Maxwell, and how quaternions led to vector analysis

### Tools

- [Browser-based robot viewer](https://robot-viewer-qfmqx.ondigitalocean.app/builder)
- [RobotDK Robot Comparator](https://robodk.com/compare-robots?d=7&rc=850&p=3.0&w=18&rp=0.100) -- compare commercial robots by reach, payload, etc.

---

## Experiment Prompts

Hands-on exercises keyed to book chapters. Each builds on the project's
notebooks and API.

| Book Chapter | Concept | Experiment |
|---|---|---|
| Ch 2 | Pose composition | Create two SE3 transforms, compose them in different orders, and verify that SE3 multiplication is non-commutative. Use `spatialmath` in `external/spatialmathematics/1`. |
| Ch 2.3 | Quaternion conversion | Convert a rotation matrix to a quaternion using `manipylator.utils.quaternion_to_rotation_matrix()` and verify the round-trip. Compare against `spatialmath.UnitQuaternion`. |
| Ch 3 | Trajectory smoothness | Generate joint trajectories with `rtb.jtraj(q_start, q_end, N)` for different values of N. Plot joint velocity and acceleration profiles and observe the quintic polynomial smoothness. |
| Ch 7.1 | Forward kinematics | Set different joint angles in `HeadlessSimulatedRobotDevice.step_to_pose()` and observe how the end-effector position changes. Compare the simulation result against `robot.model.fkine()`. |
| Ch 7.1.5 | DH parameters | Derive the DH table for a 3-link planar arm in SymPy (see `10-symbolic-manipulation.ipynb`), then build the same robot with RTB and compare `fkine()` outputs. |
| Ch 7.2 | Inverse kinematics | Pick a target end-effector position, solve with `robot.model.ikine_LM()`, then verify the solution produces the expected pose in Genesis simulation. Try different initial guesses to find multiple solutions. |
| Ch 7.3 | Cartesian trajectory | Use `manipylator.utils.parametric_heart_1(t)` to generate a heart-shaped Cartesian path, solve IK at each point, and step through the resulting joint trajectory in simulation. |
| Ch 8.1 | Jacobian | Compute `robot.model.jacob0(q)` at several configurations. Find a near-singular configuration where `det(J)` approaches zero and observe how commanded Cartesian velocities map to large joint velocities. |
| Ch 8.2 | Resolved-rate control | Work through `external/dkt/Part 1/3 Resolved-Rate Motion Control` and then try the same control law on the Empiric arm model. Observe behavior near singularities. |
| Ch 8.3 | Manipulability | Compute `robot.model.manipulability(q)` along a trajectory and plot it. Identify where the robot is most and least dexterous. |
| Ch 9 | Dynamics | Compute `robot.model.gravload(q)` at different configurations and observe which joints bear the most gravity torque. Relate this to the arm's geometry. |
| Ch 11 | Image acquisition | Use `StreamingCamera` to capture frames, then apply OpenCV filters (Gaussian blur, Canny edge detection) to a live MJPEG stream. |
| Ch 15 | Visual servoing concept | Sketch a visual servoing loop: `StreamingCamera` provides images, detect a target with OpenCV, compute the image error, map it through the image Jacobian, and send velocity commands to the robot via MQTT. (Advanced -- requires custom code.) |

---

## Simulator Demonstrations

Ready-to-run demonstrations suitable for lectures, lab introductions, or
self-study warm-ups.

| Demo | What It Shows | Book Chapter | How to Run |
|---|---|---|---|
| Genesis 3D simulation | A URDF-based robot arm responding to joint commands in real-time physics | Ch 7 | `20-simulation.ipynb` in Jupyter (lab container) |
| Headless FK sweep | Programmatically sweep joint angles and log end-effector positions without GUI overhead | Ch 7.1 | `21-headless-simulation.ipynb` |
| MQVisualizer live mirror | A Genesis window that mirrors joint state from any MQTT source in real-time | Ch 7, Ch 9 | `python run_mq_visualizer.py --broker mq` (inside lab container, with `minimal` or `simulated` profile) |
| Heart trajectory | The Empiric arm traces a heart shape in Cartesian space, solved via IK at each point | Ch 7.3, Ch 7.4 | `extra-notebooks/control-move-in-heart-shape.ipynb` |
| Trajectory analysis | Plot joint positions, velocities, and accelerations for a given trajectory using hvPlot | Ch 3, Ch 7.3 | `extra-notebooks/analyzing-trajectories-using-sympy-numpy-hvplot.ipynb` |
| Camera controls | Manipulate the Genesis scene camera programmatically (pan, zoom, orbit) | Ch 13 | `22-camera-controls.ipynb` |
| State viewer dashboard | Panel-based web dashboard showing real-time current vs target joint state | Ch 9 | `panel serve state_viewer.py` (with `minimal` profile for MQTT) |
| Safety pipeline | Hand detection triggers an e-stop event when a hand enters the camera frame | Ch 12, Ch 15 | `python run_pipeline.py --local-camera` |

---

## Lab Environment Configurations

Different classroom and lab scenarios require different levels of infrastructure.

| Scenario | Docker Profile | GPU Required | Physical Robot | Best For |
|---|---|---|---|---|
| Pure math and kinematics | `minimal` | No | No | Ch 2-3, Ch 7.1-7.2 theory. Students work with RTB, SymPy, and `spatialmath` in Jupyter. |
| Simulation and visualization | `lab` only | Yes | No | Ch 7-8 hands-on. Students use Genesis to see FK/IK/trajectories in 3D. |
| Full sim + controller | `simulated` | Yes | No | Ch 7-9, Ch 15. Simulated firmware lets students send gcode and observe state via MQTT without hardware. |
| Physical robot lab | `full` | Yes | Yes | Ch 7-9, Ch 15. Full sim-to-real pipeline. Students test in Genesis, deploy to the physical arm. |
| Vision-focused lab | `lab` + local camera | Yes | Optional | Ch 10-12, Ch 15. Students work with `StreamingCamera`, OpenCV, MediaPipe. |

---

## Assignment Ideas for Teachers

### Introductory (no GPU needed, Ch 2-3, Ch 7.1-7.2)

1. **FK verification** (Ch 7.1.5): Students derive the forward kinematics for a 3-joint planar arm using DH parameters on paper, then implement it in SymPy (see `10-symbolic-manipulation.ipynb`). They verify their result against `robot.model.fkine()` from RTB.
2. **Workspace exploration** (Ch 7.1): Using RTB, students sample random joint configurations, compute FK for each, and plot the reachable workspace of the Empiric arm. Discuss how joint limits affect the workspace.
3. **Spatial math exercises** (Ch 2): Work through `external/spatialmathematics/0..2`. Students compose rotations and translations, verify SE3 properties, and convert between representations (Euler angles, quaternions, rotation matrices).
4. **Trajectory design** (Ch 3, Ch 7.3): Students generate smooth joint-space trajectories with `rtb.jtraj()` and Cartesian trajectories with `rtb.ctraj()`. Plot position, velocity, and acceleration profiles. Discuss the quintic polynomial and why it ensures smooth motion.

### Intermediate (GPU recommended, Ch 7.2-8)

5. **IK comparison** (Ch 7.2, Ch 8.5): Students solve inverse kinematics for a target pose using both analytical and numerical methods (`ikine_LM`). Compare convergence, accuracy, and handling of multiple solutions. Verify each solution in Genesis simulation.
6. **Trajectory design project** (Ch 7.3-7.4): Design a Cartesian trajectory (e.g., writing initials, as in Ch 7.4.1), convert to joint space using IK, and execute in Genesis simulation. Evaluate smoothness and joint velocity limits.
7. **Jacobian singularities** (Ch 8.3): Identify singular configurations of the Empiric arm analytically, then demonstrate them in simulation by plotting the manipulability ellipsoid along a trajectory. Relate to the condition number of the Jacobian.
8. **Resolved-rate controller** (Ch 8.2): Implement a resolved-rate motion controller for the Empiric arm following the approach in `external/dkt/Part 1/3`. Command a straight-line Cartesian motion and observe joint velocities. Test behavior near singularities.

### Advanced (GPU required, Ch 9, Ch 15-16)

9. **Sim-to-real pipeline** (Ch 7-9): Students develop a movement sequence in headless simulation, verify it in the visual simulator, then deploy to the physical robot via the Klipper controller. Discuss sim-to-real transfer challenges.
10. **Safety system extension** (Ch 12, Ch 15): Extend the vision pipeline to detect a new object class (e.g., tools) using MediaPipe or OpenCV. Publish detection events on a new MQTT topic and integrate with the safety listener.
11. **Custom device** (Ch 8, Ch 9): Implement a new `MQClient` subclass that subscribes to robot state and computes a derived metric (e.g., manipulability index from Ch 8.3) in real time, publishing results to a custom MQTT topic.
12. **Visual servoing prototype** (Ch 15): Build a position-based visual servoing loop: use `StreamingCamera` to detect a colored target, estimate its 3D position, compute the pose error, and command the robot to approach it via `PhysicalRobotDevice` or `SimulatedRobotDevice`.
