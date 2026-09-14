import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import roboticstoolbox as rtb
import numpy as np

# Definição dos manipuladores
left_arm_charmie = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=-np.pi/2, a=0.011783, d=0,        offset=0),
        rtb.RevoluteDH(alpha=-np.pi/2, a=0,        d=0.066104, offset=-np.pi/2),
        rtb.RevoluteDH(alpha=-np.pi/2, a=0,        d=0.22460,  offset=-np.pi/2),
        rtb.RevoluteDH(alpha=np.pi/2,  a=0,        d=0,        offset=0.464),
        rtb.RevoluteDH(alpha=0,        a=0,        d=0.370,    offset=0),
    ],
    name="LeftArm"
)

right_arm = rtb.DHRobot(
    [
        rtb.RevoluteDH(alpha=-np.pi/2, a=0.011783, d=0,         offset=0),
        rtb.RevoluteDH(alpha=np.pi/2,  a=0,        d=-0.066104, offset=np.pi/2),
        rtb.RevoluteDH(alpha=np.pi/2,  a=0,        d=0.22460,   offset=np.pi/2),
        rtb.RevoluteDH(alpha=-np.pi/2, a=0,        d=0,         offset=-0.464),
        rtb.RevoluteDH(alpha=0,        a=0,        d=0.370,     offset=0),
    ],
    name="RightArmInMoov"
)

# Configuração inicial
robot = right_arm
q_init = [0.0, 0.0, 0.0, np.pi/12, 0.0]
limites_v = [-0.4, 0.8, -0.6, 0.6, -0.4, 0.6]

# Inicializa o plot nativo do RTB (sem passar fig ou ax)
env = robot.plot(q_init, block=False, backend='pyplot', limits=limites_v)

# Extrai os objetos gerenciados pelo backend do RTB
fig = env.fig
ax = env.ax

# Ajusta o layout para deixar margem para os controles
fig.subplots_adjust(bottom=0.30)
ax.view_init(elev=30, azim=45)

q_init_deg = np.rad2deg(q_init)

# Sliders para as 5 juntas
slider_axes = [fig.add_axes([0.18, 0.22 - i * 0.04, 0.65, 0.025]) for i in range(5)]
joint_names = ['J1 (Ad/Ab)', 'J2 (Rot-Ombro)', 'J3 (Rot-Braço)', 'J4 (Cotovelo)', 'J5 (Pulso)']
sliders = []

for i, (sax, name) in enumerate(zip(slider_axes, joint_names)):
    s = Slider(
        ax=sax,
        label=name,
        valmin=-90,
        valmax=90,
        valinit=q_init_deg[i],
        valstep=0.01
    )
    sliders.append(s)

# Atualização em tempo real
def update(val):
    q_current_rad = [np.deg2rad(s.val) for s in sliders]
    robot.q = q_current_rad
    env.step()

for s in sliders:
    s.on_changed(update)

# Botão de reset
reset_ax = fig.add_axes([0.85, 0.02, 0.1, 0.035])
reset_btn = Button(reset_ax, 'Reset', hovercolor='0.975')

def reset(event):
    for s in sliders:
        s.reset()

reset_btn.on_clicked(reset)

plt.show(block=True)