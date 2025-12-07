import matplotlib.pyplot as plt
import config.config as config
import numpy as np
import plotly.graph_objects as go

def plot_convergence(logbook, filename=None):
    params = config.get_problem_config()

    gen = logbook.select("gen")
    fit_max = logbook.select("max")

    real_values = [params["offset_roulette"] - f for f in fit_max]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(gen, real_values, label="Mejor Individuo", color="#1f77b4", linewidth=2)
    ax.set_title(f"Convergencia - {params['name']}", fontsize=14)
    ax.set_xlabel("Generaciones", fontsize=12)
    ax.set_ylabel("Valor de la Función", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    if filename:
        fig.savefig(filename)
        print(f"[INFO] Gráfica guardada como '{filename}'")
    elif not filename and __name__ == "__main__":
        fname = f"Convergencia_{params['name'].replace(' ', '_')}.png"
        fig.savefig(fname)
        print(f"[INFO] Gráfica guardada como '{fname}'")

    return fig

def plot_3d_surface(best_ind_decoded=None):
    """Genera un gráfico 3D interactivo para la función Camel Back."""
    if config.PROBLEM_ID != 2:
        return None 
    
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    Z = (4*X**2 - 2.1*X**4 + (1/3)*X**6) + (X*Y) + (-4*Y**2 + 4*Y**4)

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=0.8)])

    if best_ind_decoded:
        bx, by = best_ind_decoded[0], best_ind_decoded[1]
        bz = (4*bx**2 - 2.1*bx**4 + (1/3)*bx**6) + (bx*by) + (-4*by**2 + 4*by**4)
        
        fig.add_trace(go.Scatter3d(
            x=[bx], y=[by], z=[bz],
            mode='markers',
            marker={'size': 10, 'color': 'red', 'symbol': 'diamond'},
            name='Mejor Solución'
        ))

    fig.update_layout(title='Topología: Six-Hump Camel Back', autosize=True,
                      scene={"xaxis_title": "X1", "yaxis_title": "X2", "zaxis_title": "f(x)"})
    return fig

def plot_schwefel_surface_2d():
    """
    Genera una representación 3D de la función Schwefel en 2 dimensiones.
    Útil para visualizar la complejidad del terreno (muchos mínimos locales).
    """
    x = np.linspace(-500, 500, 100)
    y = np.linspace(-500, 500, 100)
    X, Y = np.meshgrid(x, y)
    
    # Fórmula Schwefel para 2 variables:
    # f(x,y) = (-x * sin(sqrt(|x|))) + (-y * sin(sqrt(|y|)))
    Z = (-X * np.sin(np.sqrt(np.abs(X)))) + (-Y * np.sin(np.sqrt(np.abs(Y))))

    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Jet', opacity=0.9)])

    fig.update_layout(
        title='Topología Referencial: Schwefel (Corte 2D)',
        autosize=False,
        width=800, height=600,
        scene={
            'xaxis_title': 'X1', 
            'yaxis_title': 'X2', 
            'zaxis_title': 'f(x)',
            'camera': {'eye': {'x': 1.5, 'y': 1.5, 'z': 0.5}}
        }
    )
    return fig