import numpy as np

N_FOLDS = 5
VOXEL_SIZE = 0.08  # aresta do cubo em metros
TEST_SIZE = 0.3


def compute_voxel_volume(pos_exp, voxel_size=VOXEL_SIZE):
    """Calcula métricas volumétricas do espaço de trabalho amostrado."""
    x_coords = pos_exp[:, 0]
    y_coords = pos_exp[:, 1]
    z_coords = pos_exp[:, 2]

    # Voxel IDs
    ix = np.floor((x_coords - x_coords.min()) / voxel_size).astype(int)
    iy = np.floor((y_coords - y_coords.min()) / voxel_size).astype(int)
    iz = np.floor((z_coords - z_coords.min()) / voxel_size).astype(int)
    voxel_ids = np.array([f"{x}_{y}_{z}" for x, y, z in zip(ix, iy, iz)])

    n_cubos = len(np.unique(voxel_ids))
    v_unitario_m3 = voxel_size**3
    v_total_m3 = n_cubos * v_unitario_m3
    v_total_litros = v_total_m3 * 1000.0

    # Volume do paralelepípedo envolvente (Bounding Box)
    dx = x_coords.max() - x_coords.min()
    dy = y_coords.max() - y_coords.min()
    dz = z_coords.max() - z_coords.min()
    v_bbox_m3 = dx * dy * dz
    taxa_ocupacao = (v_total_m3 / v_bbox_m3) * 100.0 if v_bbox_m3 > 0 else 0.0

    metricas_vol = {
        "aresta_m": voxel_size,
        "n_cubos": n_cubos,
        "volume_unitario_cm3": v_unitario_m3 * 1e6,
        "volume_ocupado_m3": v_total_m3,
        "volume_ocupado_litros": v_total_litros,
        "volume_bbox_m3": v_bbox_m3,
        "taxa_ocupacao_pct": taxa_ocupacao,
    }

    print("\n--- Métricas Volumétricas do Espaço Amostrado ---")
    print(f"• Aresta do Voxel: {voxel_size * 100:.1f} cm")
    print(f"• Cubos Únicos Ocupados: {n_cubos}")
    print(
        f"• Volume Total Ocupado: {v_total_m3:.4f} m³ ({v_total_litros:.2f} Litros)"
    )
    print(f"• Bounding Box Teórica: {v_bbox_m3:.4f} m³")
    print(f"• Densidade/Ocupação Volumétrica: {taxa_ocupacao:.2f}%")

    return metricas_vol

def create_spatial_split(
    X,
    y,
    pos_exp,
    voxel_size=VOXEL_SIZE,
    test_size=TEST_SIZE,
    n_folds=N_FOLDS,
    random_state=1,
):
    rng = np.random.RandomState(random_state)

    x_coords = pos_exp[:, 0]
    y_coords = pos_exp[:, 1]
    z_coords = pos_exp[:, 2]

    # 1. Identificação discreta do cubo de cada ponto
    ix = np.floor((x_coords - x_coords.min()) / voxel_size).astype(int)
    iy = np.floor((y_coords - y_coords.min()) / voxel_size).astype(int)
    iz = np.floor((z_coords - z_coords.min()) / voxel_size).astype(int)

    voxel_ids = np.array([f"{x}_{y}_{z}" for x, y, z in zip(ix, iy, iz)])
    print(f"Número de cubos ocupados: {len(np.unique(voxel_ids))}")

    # Agrupa índices por voxel
    voxels = {}
    for idx, vid in enumerate(voxel_ids):
        voxels.setdefault(vid, []).append(idx)

    idx_dev = []
    idx_test = []

    # 2. Divisão Estratificada: Desenvolvimento (Treino+Val) vs Teste
    # Amostramos DENTRO de cada cubo
    test_counter = 0
    for vid, idx_list in voxels.items():
        idx_list = np.array(idx_list)
        rng.shuffle(idx_list)
        n = len(idx_list)

        if n == 1:
            # Cubo com apenas 1 ponto: vai para desenvolvimento para garantir cobertura
            idx_dev.append(idx_list[0])
        else:
            n_test = int(np.round(n * test_size))
            if n_test == 0 and n >= 4:
                n_test = 1
            if n_test == 0 and test_counter % 3 == 0:
                n_test = 1
                test_counter += 1

            n_dev = n - n_test
            idx_dev.extend(idx_list[:n_dev])
            idx_test.extend(idx_list[n_dev:])

    idx_dev = np.array(idx_dev)
    idx_test = np.array(idx_test)

    # 3. K-Fold Estratificado dentro do conjunto de desenvolvimento
    # Agrupa os índices de dev novamente por voxel
    dev_voxels = {}
    for idx in idx_dev:
        vid = voxel_ids[idx]
        dev_voxels.setdefault(vid, []).append(idx)

    # Atribuição cíclica de dobras (0 a n_folds-1) intra-voxel
    fold_assignment = {}
    k_counter = 0

    for vid, idx_list in dev_voxels.items():
        idx_list = np.array(idx_list)
        rng.shuffle(idx_list)
        for local_idx in idx_list:
            fold_assignment[local_idx] = k_counter % n_folds
            k_counter += 1

    folds = []
    for f in range(n_folds):
        val_idx = np.array([i for i in idx_dev if fold_assignment[i] == f])
        train_idx = np.array([i for i in idx_dev if fold_assignment[i] != f])

        folds.append({"fold": f + 1, "train_idx": train_idx, "val_idx": val_idx})

        print(f"Fold {f+1}: {len(train_idx)} treino | {len(val_idx)} validação")

    return {
        "idx_dev": idx_dev,
        "idx_test": idx_test,
        "folds": folds,
        "voxel_ids": voxel_ids,
    }