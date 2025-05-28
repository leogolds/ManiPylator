# ManiPylator
<img src="logo.webp" width="150" height="150" align="right">

3D printed 6DOF robotic manipulator powered by your favorite snake based programming language


## Lab Environment
Spin up a jupyter lab environment:
``` bash
docker compose up -d
```

To allow the jupyter container to open the genesis window, run:
``` bash
xhost +local:root
```

Go to [http://localhost:8888/lab/workspaces/auto-0/tree/generate-trajectory-example.ipynb](http://localhost:8888/lab/workspaces/auto-0/tree/generate-trajectory-example.ipynb) for a short intro. The notebook demonstrats loading a URDF, forward/inverse kinematics, and saving control signals as a csv.