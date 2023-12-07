# Robot_Arm_Balance_Pendulum

In this repository we provide the code to our final project for the M2 MVA's 2023 Robotics course, which solves the Pendulum balancing problem over a UR5 arm descriptor using optimal control.

There are different files for the different parts of the project:
- Part 2: `Cartpole_LQR.ipynb`. Using LQR to balance a pendulum using Gym's Cartpole environment.
- Part 4: `Robotics_arms.ipnyb`. Using a PD controller to perform forward kinematics with a UR5 arm descriptor.
- Part 5: `ee_line_motion.ipynb`. Using inverse kinematics to restrict the movement of the arm to a horizontal line.
- Part 6: `ur5_modified_robot.urdf`. Modifying the URDF file for the arm descriptor in order to add a pendulum.
- Part 7: `observation.py`. Adaptation of our solution to Part 2 with the whole arm description. 
