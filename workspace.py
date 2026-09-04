import numpy as np
import matplotlib.pyplot as plt
import roboticstoolbox as rtb

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

N = 20000

q1 = np.random.uniform(-np.pi/2, np.pi/2, N)
q2 = np.random.uniform(-np.pi/4, 0, N)
q3 = np.random.uniform(-np.pi/2, np.pi/2, N)
q4 = np.random.uniform(-np.pi/2, np.pi/2, N)
q5 = np.random.uniform(-np.pi/4, 0, N)

q_rand = np.column_stack((q1, q2, q3, q4, q5))

T = right_arm.fkine(q_rand)

posicoes = T.t

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# O parâmetro 's' é o tamanho do ponto, 'alpha' deixa transparente para ver a densidade
ax.scatter(posicoes[:, 0], posicoes[:, 1], posicoes[:, 2], s=1, c='blue', alpha=0.5)

ax.set_title('Espaço de Trabalho - Braço Direito do InMoov')
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_zlabel('Z (m)', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.view_init(elev=0, azim=90)
# Força os eixos a manterem proporção igual (para não distorcer o braço)
ax.set_box_aspect([1, 1, 1])

plt.show()