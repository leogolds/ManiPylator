#!/usr/bin/env python3
"""
ManiPylator System Design Diagram
Comprehensive architecture of the open-source robotics education platform
"""

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.storage import S3
from diagrams.generic.compute import Rack
from diagrams.generic.database import SQL
from diagrams.generic.device import Mobile, Tablet
from diagrams.generic.storage import Storage
from diagrams.oci.compute import VM
from diagrams.onprem.compute import Server
from diagrams.onprem.container import Docker
from diagrams.onprem.queue import Rabbitmq
from diagrams.programming.framework import Django
from diagrams.programming.language import JavaScript, Python

with Diagram("ManiPylator: Open-Source Robotics Education Platform",
             filename="manipylator_architecture",
             show=False,
             direction="TB"):

    # Entry Point - Students/Educators/Makers
    with Cluster("👨‍🎓 Learning Community"):
        students = Mobile("Students")
        educators = Tablet("Educators")
        makers = Mobile("Makers")
        researchers = Tablet("Researchers")

    # Web Interface Layer
    with Cluster("🌐 Web Interface"):
        browser = JavaScript("Web Browser\n(localhost:8888)")

    # Containerized Development Environment
    with Cluster("🐳 Docker Environment"):
        docker_compose = Docker("Docker Compose\n(GPU + X11)")
        jupyter_lab = Python("JupyterLab\n(Interactive IDE)")

    # Educational Curriculum - Progressive Learning Path
    with Cluster("📚 4-Phase Curriculum"):
        with Cluster("Phase 1: Math Foundations"):
            math_rotations = Python("rotations-sympy-rtb.ipynb")
            math_trajectories = Python("analyzing-trajectories.ipynb")
            math_rotations >> math_trajectories

        with Cluster("Phase 2: Robot Kinematics"):
            traj_gen = Python("generate-trajectory.ipynb")
            kinematics = Python("kinematics-example.ipynb")
            heart_control = Python("control-heart-shape.ipynb")
            traj_gen >> kinematics >> heart_control

        with Cluster("Phase 3: Path Planning"):
            simple_control = Python("simplified-control-*.ipynb")
            manual_control = Python("manual-control.ipynb")
            measured_control = Python("measured-control.ipynb")
            simple_control >> manual_control >> measured_control

        with Cluster("Phase 4: Integration"):
            comms = Python("comms-control.ipynb")
            visualizer = Python("mq-visualizer.ipynb")
            dashboard = Python("panel-genesis-v2.ipynb")
            comms >> visualizer >> dashboard

    # Core Technology Stack - Simulation & Computation
    with Cluster("⚡ Simulation Engine"):
        genesis = Server("Genesis Physics\n(GPU Accelerated)")
        robotics_tb = Python("Robotics Toolbox\n(Kinematics/Dynamics)")
        spatial_math = Python("SpatialMath\n(3D Transforms)")
        numpy_scipy = Python("NumPy/SciPy\n(Math Computing)")

        genesis >> robotics_tb
        robotics_tb >> spatial_math
        spatial_math >> numpy_scipy

    # Interactive Visualization & Control
    with Cluster("📊 Interactive Visualization"):
        panel_framework = Python("Panel Framework\n(Interactive Apps)")
        bokeh_holoviews = JavaScript("Bokeh + HoloViews\n(Web Viz)")
        hvplot = Python("hvPlot\n(High-level Plotting)")

        panel_framework >> bokeh_holoviews
        hvplot >> bokeh_holoviews

    # Robot Model & Configuration
    with Cluster("🤖 Robot Definition"):
        urdf_model = Storage("URDF Models\n(Robot Description)")
        stl_meshes = S3("3D Meshes\n(Visual/Collision)")
        robot_config = SQL("Configuration\n(Parameters)")

        urdf_model >> stl_meshes
        urdf_model >> robot_config

    # Communication & Control Layer
    with Cluster("📡 Communication Stack"):
        mqtt_broker = Rabbitmq("MQTT Broker\n(Wireless Comms)")
        moonraker_api = Server("Moonraker API\n(3D Printer Interface)")

        mqtt_broker >> moonraker_api

    # Physical Hardware Path
    with Cluster("🔧 Physical Robot"):
        klipper_fw = Server("Klipper Firmware\n(Real-time Control)")
        stm32_mcu = Rack("STM32 MCU\n(Microcontroller)")
        stepper_motors = Rack("6x Stepper Motors\n(Joint Actuators)")
        printed_robot = Rack("3D Printed Robot\n(6DOF Manipulator)")

        klipper_fw >> stm32_mcu
        stm32_mcu >> stepper_motors
        stepper_motors >> printed_robot

    # Simulation Alternative Path
    with Cluster("💻 Virtual Hardware"):
        simulavr = VM("SimulAVR\n(MCU Simulator)")
        virtual_robot = Django("Virtual Robot\n(Genesis Simulation)")

        simulavr >> virtual_robot

    # Main Data Flows
    students >> browser
    educators >> browser
    makers >> browser
    researchers >> browser

    browser >> docker_compose
    docker_compose >> jupyter_lab

    # Educational progression flow
    jupyter_lab >> math_rotations
    math_trajectories >> traj_gen
    heart_control >> simple_control
    measured_control >> comms
    dashboard >> Edge(label="Complete", style="bold", color="green") >> students

    # Technology integration
    jupyter_lab >> genesis
    jupyter_lab >> robotics_tb
    jupyter_lab >> panel_framework

    # Robot model usage
    urdf_model >> genesis
    urdf_model >> robotics_tb

    # Visualization pipeline
    genesis >> panel_framework
    robotics_tb >> hvplot
    panel_framework >> browser

    # Communication flows
    dashboard >> mqtt_broker
    moonraker_api >> klipper_fw
    moonraker_api >> simulavr

    # Hardware control
    klipper_fw >> printed_robot
    simulavr >> virtual_robot

    # Feedback loops
    printed_robot >> Edge(label="Real Feedback", style="dashed", color="green") >> students
    virtual_robot >> Edge(label="Sim Feedback", style="dashed", color="blue") >> students

print("🎯 ManiPylator System Architecture Generated!")
print("📊 Output: manipylator_architecture.png")
print("\n🔑 Key Educational Features:")
print("  • Progressive 4-phase curriculum")
print("  • Simulation-to-hardware pipeline")
print("  • Interactive visualization & control")
print("  • Open-source transparency")
print("  • Containerized reproducibility")
print("  • Real-world maker accessibility")
