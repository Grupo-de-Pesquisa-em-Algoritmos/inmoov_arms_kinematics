import numpy as np
import plotly.graph_objects as go


def get_voxel_information(pos_exp, voxel_size):
    """Calcula o ID e as coordenadas inteiras de voxel para cada ponto."""
    x_coords = pos_exp[:, 0]
    y_coords = pos_exp[:, 1]
    z_coords = pos_exp[:, 2]

    x_min = x_coords.min()
    y_min = y_coords.min()
    z_min = z_coords.min()

    ix = np.floor((x_coords - x_min) / voxel_size).astype(int)
    iy = np.floor((y_coords - y_min) / voxel_size).astype(int)
    iz = np.floor((z_coords - z_min) / voxel_size).astype(int)

    voxel_ids = np.array([f"{x}_{y}_{z}" for x, y, z in zip(ix, iy, iz)])
    voxel_coords = np.column_stack([ix, iy, iz])

    return voxel_ids, voxel_coords, (x_min, y_min, z_min)


def classify_voxels_stratified(voxel_ids, idx_test, folds, fold_to_show):
    """
    Classifica cada cubo sem sobrescrita:
    - Se o cubo possui pontos de mais de uma partição, classifica como 'Estratificado'.
    - Se possui pontos de apenas uma partição, classifica como o nome correspondente.
    """
    fold_data = next(f for f in folds if f["fold"] == fold_to_show)
    set_train = set(fold_data["train_idx"])
    set_val = set(fold_data["val_idx"])
    set_test = set(idx_test)

    voxel_contents = {}
    for idx, vid in enumerate(voxel_ids):
        voxel_contents.setdefault(vid, set())
        if idx in set_test:
            voxel_contents[vid].add("Teste")
        elif idx in set_val:
            voxel_contents[vid].add("Validação")
        elif idx in set_train:
            voxel_contents[vid].add("Treino")

    classification = {}
    for vid, sets_present in voxel_contents.items():
        if len(sets_present) > 1:
            classification[vid] = "Estratificado"
        elif "Teste" in sets_present:
            classification[vid] = "Teste"
        elif "Validação" in sets_present:
            classification[vid] = "Validação"
        else:
            classification[vid] = "Treino"

    return classification


def create_cube_vertices(ix, iy, iz, voxel_size, x_min, y_min, z_min):
    """Retorna as coordenadas dos 8 vértices do cubo tridimensional."""
    x0 = x_min + ix * voxel_size
    x1 = x0 + voxel_size
    y0 = y_min + iy * voxel_size
    y1 = y0 + voxel_size
    z0 = z_min + iz * voxel_size
    z1 = z0 + voxel_size

    return np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ])


def create_cube_mesh():
    """Retorna os índices triangulares que formam as 12 faces do cubo."""
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    return i, j, k


def plot_spatial_voxels(pos_exp, idx_test, folds, voxel_size=0.08, fold_to_show=1):
    """Gera visualização 3D com cubos delimitadores e pontos das amostras."""
    voxel_ids, voxel_coords, mins = get_voxel_information(pos_exp, voxel_size)
    x_min, y_min, z_min = mins

    classification = classify_voxels_stratified(
        voxel_ids, idx_test, folds, fold_to_show
    )

    fig = go.Figure()

    colors_cubes = {
        "Estratificado": "rgba(160, 160, 160, 0.08)",  # cinza translúcido
        "Treino": "rgba(31, 119, 180, 0.12)",         # azul translúcido
        "Validação": "rgba(255, 127, 14, 0.12)",       # laranja translúcido
        "Teste": "rgba(214, 39, 40, 0.15)"             # vermelho translúcido
    }

    # 1. Renderização das caixas dos cubos (Voxels)
    i_tri, j_tri, k_tri = create_cube_mesh()
    plotted_cube_labels = set()
    unique_voxels = np.unique(voxel_coords, axis=0)

    for voxel in unique_voxels:
        ix, iy, iz = voxel
        vid = f"{ix}_{iy}_{iz}"
        if vid not in classification:
            continue

        label = classification[vid]
        vertices = create_cube_vertices(ix, iy, iz, voxel_size, x_min, y_min, z_min)
        show_leg = label not in plotted_cube_labels

        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i_tri,
                j=j_tri,
                k=k_tri,
                color=colors_cubes[label],
                name=f"Voxel {label}",
                legendgroup=f"Cubo {label}",
                showlegend=show_leg,
                hoverinfo="skip"
            )
        )
        plotted_cube_labels.add(label)

    # 2. Renderização dos Pontos Experimentais por Partição
    fold_data = next(f for f in folds if f["fold"] == fold_to_show)
    idx_train = fold_data["train_idx"]
    idx_val = fold_data["val_idx"]

    # Pontos de Treino
    fig.add_trace(
        go.Scatter3d(
            x=pos_exp[idx_train, 0],
            y=pos_exp[idx_train, 1],
            z=pos_exp[idx_train, 2],
            mode="markers",
            name=f"Treino ({len(idx_train)} amostras)",
            marker=dict(size=2.5, color="#1f77b4", opacity=0.55),
            hovertemplate="Treino<br>X: %{x:.3f} m<br>Y: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>"
        )
    )

    # Pontos de Validação
    fig.add_trace(
        go.Scatter3d(
            x=pos_exp[idx_val, 0],
            y=pos_exp[idx_val, 1],
            z=pos_exp[idx_val, 2],
            mode="markers",
            name=f"Validação ({len(idx_val)} amostras)",
            marker=dict(size=3.8, color="#ff7f0e", opacity=0.85),
            hovertemplate="Validação<br>X: %{x:.3f} m<br>Y: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>"
        )
    )

    # Pontos de Teste (Holdout Cego)
    fig.add_trace(
        go.Scatter3d(
            x=pos_exp[idx_test, 0],
            y=pos_exp[idx_test, 1],
            z=pos_exp[idx_test, 2],
            mode="markers",
            name=f"Teste Cego ({len(idx_test)} amostras)",
            marker=dict(size=4.0, color="#d62728", opacity=0.9),
            hovertemplate="Teste Cego<br>X: %{x:.3f} m<br>Y: %{y:.3f} m<br>Z: %{z:.3f} m<extra></extra>"
        )
    )

    # 3. Layout e Proporção Física 1:1:1
    fig.update_layout(
        title=dict(
            text=f"Amostragem Estratificada por Voxels — Fold {fold_to_show}",
            x=0.05,
            y=0.96,
            font=dict(size=16)
        ),
        width=900,
        height=800,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.6, y=-1.6, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        ),
        legend=dict(
            title=dict(text="Partições"),
            yanchor="top",
            y=0.93,
            xanchor="left",
            x=0.03,
            bgcolor="rgba(255, 255, 255, 0.75)",
            itemsizing="constant"
        ),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    return fig