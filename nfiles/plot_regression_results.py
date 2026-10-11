import numpy as np
import plotly.graph_objects as go


def plot_real_vs_pred(y_real, y_pred, model_name="Melhor Modelo"):
    """
    Plota o gráfico interativo de Valor Previsto vs. Valor Real com Plotly,
    incluindo a linha ideal y = y_pred e métricas de erro.
    """
    y_real = np.asarray(y_real)
    y_pred = np.asarray(y_pred)

    min_val = min(y_real.min(), y_pred.min())
    max_val = max(y_real.max(), y_pred.max())

    # Cálculo dos resíduos (|Real - Previsto|)
    residuos = np.abs(y_real - y_pred)

    fig = go.Figure()

    # 1. Linha Ideal de 45 graus (Previsão Perfeita)
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Ideal (y = ŷ)",
            line=dict(color="crimson", dash="dash", width=2),
            hoverinfo="skip"
        )
    )

    # 2. Amostras de Teste
    fig.add_trace(
        go.Scatter(
            x=y_real,
            y=y_pred,
            mode="markers",
            name="Amostras de Teste",
            marker=dict(
                size=7,
                color=residuos,
                colorscale="Viridis",
                colorbar=dict(title="|Resíduo| (J)"),
                opacity=0.75,
                line=dict(color="black", width=0.5)
            ),
            hovertemplate=(
                "<b>Amostra de Teste</b><br>"
                "Real: %{x:.2f} J<br>"
                "Previsto: %{y:.2f} J<br>"
                "Erro Absoluto: %{marker.color:.2f} J"
                "<extra></extra>"
            )
        )
    )

    # 3. Formatação do Layout
    fig.update_layout(
        title=dict(
            text=f"Acurácia no Teste Cego: Valor Real vs. Previsto — {model_name}",
            x=0.05,
            y=0.96,
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Consumo Energético Real (J)",
            range=[min_val - 2, max_val + 2],
            showgrid=True
        ),
        yaxis=dict(
            title="Consumo Energético Previsto (J)",
            range=[min_val - 2, max_val + 2],
            showgrid=True,
            scaleanchor="x",  # Mantém proporção 1:1 entre eixos X e Y
            scaleratio=1
        ),
        width=750,
        height=700,
        legend=dict(
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=0.05,
            bgcolor="rgba(255, 255, 255, 0.7)"
        ),
        margin=dict(l=50, r=50, b=50, t=60)
    )

    return fig