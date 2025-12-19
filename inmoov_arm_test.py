import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sp
import random

# creation of the left arm based on inmoov's DH parameters
left_arm = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=(np.pi)/2, offset=-(np.pi)/2),
        rtb.RevoluteDH(d=-0.18, alpha=np.pi/2, offset=np.pi/2),
        rtb.RevoluteDH(d=0.286, alpha=-(np.pi)/2, offset=-(np.pi)/2),
        rtb.RevoluteDH(d=0.0135, a=0.0127, alpha=np.pi/2),
        rtb.RevoluteDH(d=0.28)
    ],
    name="LeftArm"
)

# (fixed) base's position adjustment (fixed)
left_arm.base = sp.SE3(0.2, 0, 0) * sp.SE3.Rx(np.pi/2)

b = left_arm.base.t

# generating a random angles vector between -90º and 90º for the five joints
angles = [random.uniform(-np.pi/2, np.pi/2) for _ in range(5)]
# angles = [0, np.pi/2, np.pi/4, 0, 0]
print(angles)
forward = left_arm.fkine(angles)
print("forwardk = \n", forward)
Y = sp.SE3(0.3, 0.2, -0.1)
T = left_arm.ikine_LM(forward)
print(T)
left_arm.plot(T.q, backend='pyplot', block=False)
ax = plt.gca()
ax.scatter(b[0], b[1], b[2], s=100, c='red', label='base')
ax.scatter(forward.t[0], forward.t[1], forward.t[2], s=80, c='blue', label='goal')
ax.legend()
# plt.scatter(b[0], b[1], b[2], s=100, c='red', label='Base')
# plt.legend()
plt.show(block=True)
