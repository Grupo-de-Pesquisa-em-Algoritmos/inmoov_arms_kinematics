import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
import roboticstoolbox as rtb
import numpy as np


# Define a robot using DH parameters (example: Puma560)
robot = rtb.models.DH.Puma560()
print(robot)
# Define a desired end-effector pose (e.g., identity matrix for home position)
T_desired = robot.fkine(robot.qr) # Forward kinematics to get a target pose

# Solve inverse kinematics
q_solution = robot.ikine_LM(T_desired) # Levenberg-Marquardt method
print(q_solution)
# robot.plot(q_solution.q)
# plt.show(block=True)

print(robot.base)

