# @title
import roboticstoolbox as rtb
import plotly.graph_objects as go
import spatialmath as sp
import numpy as np
from data_utils import read_dataset
FILENAME = "dataset_regressao_energia.csv"

# ==============================================================================
# ÂNGULOS LIMITES PARA O BRAÇO REAL E VIRTUAL
# ==============================================================================
# definicao de angulos minimos e maximos para o ambiente virtual
angles_min_v = np.array([0, -90, -90, 35, -90])
angles_max_v = np.array([-45, 90, 90, 90, 90])


# definicao de angulos minimos e maximos para o ambiente real
angles_min_r = np.array([85, 0, 0, 180, 0])
angles_max_r = np.array([30, 180, 180, 125, 180])
# ==============================================================================


# ==============================================================================
# DEFINIÇÃO DO BRAÇO VIRTUAL A PARTIR DOS PARÂMETROS DE DH (Syed et al 2024).
# ==============================================================================
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
right_arm.base = sp.SE3(0, 0.25, 0) * sp.SE3.Rx(np.pi/2)


def get_eff_pos(X, y):
    # ==============================================================================
    # OBTENÇÃO DADOS EXPERIMENTAIS E FK
    # ==============================================================================
    colunas_servos = [f"servo{i}_angulo_final" for i in range(5, 0, -1)]
    q_real = X[colunas_servos].to_numpy()

    # traducao dos angulos reais para correspondentes no amb virtual
    q_virtual = angles_min_v + ((q_real - angles_min_r) / (angles_max_r - angles_min_r)) * (angles_max_v - angles_min_v)

    T = right_arm.fkine(np.radians(q_virtual))
    pos_exp = T.t
    return pos_exp

def translate_real_to_virtual(q_r, min_r, max_r, min_v, max_v):
    q_r = np.asarray(q_r)
    min_r, max_r = np.asarray(min_r), np.asarray(max_r)
    min_v, max_v = np.asarray(min_v), np.asarray(max_v)

    return min_v + ((q_r - min_r) / (max_r - min_r)) * (max_v - min_v)

X, y = read_dataset(FILENAME)
pos_exp = get_eff_pos(X, y)


# ==============================================================================
# GERAÇÃO DE PONTOS RANDÔMICOS DO WORKSPACE
# ==============================================================================
# quantidade de pontos aleatórios para serem gerados do workspace
N_teorico = 20000

# Garante o menor e maior valor por junta (corrige a junta 1 onde min > max numericamente)
lim_inf = np.minimum(angles_min_v, angles_max_v)
lim_sup = np.maximum(angles_min_v, angles_max_v)

# Amostragem uniforme no espaço de juntas
q_teorico = np.random.uniform(lim_inf, lim_sup, size=(N_teorico, 5))

# Cinemática direta teórica
T_teo = right_arm.fkine(np.radians(q_teorico))
pos_teo = T_teo.t

# ==============================================================================

"""
# ==============================================================================
# PLOT COMPARATIVO DOS DADOS EXPERIMENTAIS E WORKSPACE TEORICO
# ==============================================================================
fig = go.Figure()

# Espaço Teórico
fig.add_trace(
    go.Scatter3d(
        x=pos_teo[:, 0],
        y=pos_teo[:, 1],
        z=pos_teo[:, 2],
        mode="markers",
        name="Espaço Teórico (" + str(int(N_teorico/1000)) + "k)",
        marker=dict(
            size=2,
            color="coral",
            opacity=0.3,
        ),
    )
)

# Pontos Experimentais Alcançados
fig.add_trace(
    go.Scatter3d(
        x=pos_exp[:, 0],
        y=pos_exp[:, 1],
        z=pos_exp[:, 2],
        mode="markers",
        name="Experimentos Reais",
        marker=dict(
            size=2.5,
            color="darkcyan",
            opacity=0.7,
        ),
    )
)

# Ajustes de Layout e Exibição
fig.update_layout(
    title="Comparação: Espaço de Trabalho Teórico vs. Experimentos Reais",
    width=800,
    height=800,
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data",  # Mantém as proporções reais do robô sem distorção

    ),
    #scene_camera=dict(
    #    eye=dict(x=0, y=-2.5, z=0),
    #    up=dict(x=0, y=0, z=1),
    #    projection=dict(type="orthographic"),
    #), # visao XZ
    legend=dict(
        yanchor="top",
        y=0.95,
        xanchor="left",
        x=0.05,
        itemsizing="constant",  # Mantém os ícones da legenda visíveis
        font=dict(
            size=14,
        )

    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

fig.show()
pos_exp
"""