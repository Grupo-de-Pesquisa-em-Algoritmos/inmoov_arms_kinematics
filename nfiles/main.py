from plot_experiments import pos_exp
from regression_model import (
    idx_dev,
    idx_test,
    folds,
)
from data_split import VOXEL_SIZE
from train_models import executar_benchmark
from visualize_split import plot_spatial_voxels
from plot_regression_results import plot_real_vs_pred


fig = plot_spatial_voxels(
    pos_exp=pos_exp,
    idx_test=idx_test,
    folds=folds,
    voxel_size=VOXEL_SIZE,
    fold_to_show=5
)

fig.show()

resultados = executar_benchmark()

fig_pred = plot_real_vs_pred(
    y_real=resultados["y_test"],
    y_pred=resultados["y_pred_best"],
    model_name=resultados["melhor_nome"]
)

fig_pred.show()