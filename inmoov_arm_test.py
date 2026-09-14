import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import numpy as np
import spatialmath as sp
import math as m 
import random

def translate_angle(min_v, max_v, min_r, max_r, angles_v):
    trans_angles = []
    for i, angle in enumerate(angles_v):
        trans_angle = (((angle-min_v[i])*(max_r[i]-min_r[i]))/(max_v[i]-min_v[i])) + min_r[i]
        #trans_angle = min_r[i] + ((angle - min_v[i])/(max_v[i]-min_v[i])) * (max_r[i] - min_r[i])
        trans_angles.append(trans_angle)

    return trans_angles



env = rtb.backends.PyPlot.PyPlot()
env.launch()

# dh parameters presented in syed et al article (2024) ----> currently preferred
left_arm_charmie = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=-(np.pi)/2,  a=0.011783, d=0,             offset=0),
        rtb.RevoluteDH(alpha=-(np.pi)/2,  a=0,        d=0.066104,      offset=-(np.pi)/2),
        rtb.RevoluteDH(alpha=-(np.pi)/2,  a=0,        d=0.22460,       offset=-(np.pi)/2),
        rtb.RevoluteDH(alpha=(np.pi)/2,   a=0,        d=0,             offset=0.464),
        rtb.RevoluteDH(alpha=0,           a=0,        d=0.370,         offset=0),
    ],
    name="LeftArm"
)

# adaptation of syed et al (24) for the right arm
right_arm = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=-(np.pi)/2,  a=0.011783, d=0,             offset=0),
        rtb.RevoluteDH(alpha=(np.pi)/2,  a=0,        d=-0.066104,      offset=(np.pi)/2),
        rtb.RevoluteDH(alpha=(np.pi)/2,  a=0,        d=0.22460,       offset=(np.pi)/2),
        rtb.RevoluteDH(alpha=-(np.pi)/2,   a=0,        d=0,             offset=-0.464),
        rtb.RevoluteDH(alpha=0,           a=0,        d=0.370,         offset=0),
    ],
    name="RightArmInMoov"
)

angles = np.array([[0, 0, 0, 0, 0], [0, -np.pi/8, 0, 0, 0], [0, -np.pi/4, 0, 0, 0]])
angle = [0, np.pi/4, 0, 0, 0]
            # ad/ab, rot-omb, rot-bra, bic, pulso
rest_angle = [0, 0, 0, np.pi/12, 0]
zero_angle = np.zeros(5)

# trajectory test angles ----------------------------------
angles1 = [0, 0, 0, 0, -np.pi]
angles2r = [-np.pi/4, 0, 0, 0, np.pi]
angles2l = [-np.pi/4, 0, 0, 0, 0]
# ---------------------------------------------------------


test_angles = [0, -(np.pi)/4, -(np.pi)/2, 0.464, 0]
# (fixed) base's position adjustment (fixed)
# left_arm.base = sp.SE3(0.2, 0, 0) * sp.SE3.Rx(np.pi/2)

b = left_arm_charmie.base.t

# definicao de angulos minimos e maximos para o ambiente virtual
angles_min_v = [0, -90, -90, 35, -90]
angles_max_v = [-45, 90, 90, 90, 90]


# definicao de angulos minimos e maximos para o ambiente real
angles_min_r = [85, 0, 0, 180, 0]
angles_max_r = [30, 180, 180, 125, 180]

angles_v = [85, 0, 0, 180, 0]

ta = translate_angle(angles_min_r, angles_max_r, angles_min_v, angles_max_v, angles_v)
print(ta)

limites_v = [-0.2, 0.8, -0.2, 0.2, -0.2, 0.2]
#right_arm.plot(np.radians(angles_min_v), block=False, backend='pyplot', limits=limites_v)
ax = plt.gca()
# frontal = [elev=0, azim=0]
# superior = [elev=0, azim=-90]
# lateral = [elev=90, azim=0]
ax.view_init(elev=90, azim=0)
#plt.draw()
#plt.show(block=True)

right_arm.base = sp.SE3(0, 0.25, 0) * sp.SE3.Rx(np.pi/2)
env.add(right_arm)
#env.add(left_arm_charmie)

qtr = rtb.jtraj(np.radians(angles_min_v), np.radians(angles_max_v), 70)
qtl = rtb.jtraj(angles1, angles2l, 50)

for q_r, q_l in zip(qtr.q, qtl.q):
    #left_arm_charmie.q = q_l
    right_arm.q = q_r
    env.step()

env.hold()

#forward = right_arm.fkine(test_angles)
#print(forward)
#forward = left_arm_charmie.fkine(test_angles)
#print(forward)
#plt.show(block=True)
"""
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
"""