import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sp
import random

# creation of the left arm based on inmoov's DH parameters
left_arm_dh = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=(np.pi)/2, offset=-(np.pi)/2),
        rtb.RevoluteDH(alpha=np.pi/2, d=-0.18, offset=np.pi/2),
        rtb.RevoluteDH(d=0.286, alpha=-(np.pi)/2, offset=-(np.pi)/2),
        rtb.RevoluteDH(d=0.0135, a=0.0127, alpha=np.pi/2),
        rtb.RevoluteDH(d=0.28)
    ],
    name="LeftArm"
)


# dh parameters presented in syed et al article (2024)
left_arm_charmie = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=-(np.pi)/2,  a=11.783,    d=0,     offset=0, flip=False),
        rtb.RevoluteDH(alpha=-(np.pi)/2, a=0,   d=66.104,      offset=-(np.pi)/2),
        rtb.RevoluteDH(alpha=-(np.pi)/2,  a=0,      d=224.60,  offset=-(np.pi)/2),
        rtb.RevoluteDH(alpha=(np.pi)/2, a=0,      d=0,  offset=0.464),
        rtb.RevoluteDH(alpha=0,  a=0, d=370, offset=0),
    ],
    name="LeftArm"
)

# the version below is the one presented on Kenshimov's article
left_arm = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=(np.pi)/2,  a=0.2,    d=0,     offset=0),
        rtb.RevoluteDH(alpha=-(np.pi)/2, a=0.05,   d=0,      offset=-(np.pi)/2),
        rtb.RevoluteDH(alpha=(np.pi)/2,  a=0,      d=-0.18,  offset=(np.pi)/2),
        rtb.RevoluteDH(alpha=-(np.pi)/2, a=0,      d=0.286,  offset=-(np.pi)/2),
        rtb.RevoluteDH(alpha=(np.pi)/2,  a=0.0127, d=0.0135, offset=0),
        rtb.RevoluteDH(alpha=0,          a=0,      d=0.28,   offset=0)
    ],
    name="LeftArm"
)

# 4dof version of the inmoov arm by Abdelaziz
# (doesn't include the wrist)
arm = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=np.pi/2, a=152.4, d=39.4),
        rtb.RevoluteDH(alpha=np.pi/2, a=180.6, d=0),
        rtb.RevoluteDH(alpha=-(np.pi)/2, a=282.6, d=28.4),
        rtb.RevoluteDH(alpha=np.pi/2, a=266.7, d=12.7),
        rtb.RevoluteDH(alpha=0, a=228.6, d=0)
    ]
)

angles = np.array([[0, 0, 0, 0, 0], [0, -np.pi/8, 0, 0, 0], [0, -np.pi/4, 0, 0, 0]])
angle = [0, np.pi/4, 0, 0, 0]
# (fixed) base's position adjustment (fixed)
# left_arm.base = sp.SE3(0.2, 0, 0) * sp.SE3.Rx(np.pi/2)

b = left_arm_charmie.base.t

# generating a random angles vector between -90º and 90º for the five joints
angles_fk = [random.uniform(-np.pi/2, np.pi/2) for _ in range(5)]
# getting the fk for the random configuration 
forward = left_arm_charmie.fkine(angles_fk)
print("random config = \n" + str(forward))
# Y = sp.SE3(0.3, 0.2, -0.1)
# using the random config from the previous line to simulate the ik calculus
T = left_arm_charmie.ikine_LM(forward)
print("results of ikine_LM = \n" + str(T))

# plot of the results (schematics of the arm + highlighted base and goal)
left_arm_charmie.plot(T.q, block=False, backend='pyplot')
ax = plt.gca()
ax.scatter(b[0], b[1], b[2], s=100, c='red', label='base')
ax.scatter(forward.t[0], forward.t[1], forward.t[2], s=80, c='blue', label='goal')
ax.legend()
plt.legend()
plt.show(block=True)
