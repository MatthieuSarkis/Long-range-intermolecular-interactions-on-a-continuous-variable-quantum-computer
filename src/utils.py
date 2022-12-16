import matplotlib.pyplot as plt
import numpy as np

def plot_partial_wigner_function(
    state,
    mode
):

    fig = plt.figure()
    X = np.linspace(-5, 5, 100)
    P = np.linspace(-5, 5, 100)
    Z = state.wigner(mode, X, P)
    X, P = np.meshgrid(X, P)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, P, Z, cmap="RdYlGn", lw=0.5, rstride=1, cstride=1)
    fig.set_size_inches(4.8, 5)
    ax.set_axis_off()
    plt.show()

def plot_loss_history(
    loss_history,
    save_path: str
) -> None:

    plt.style.use('./src/plots.mplstyle')
    fig, axes = plt.subplots(nrows=1, ncols=1)
    axes.plot(loss_history)
    axes.set_xlabel('Epoch')
    axes.set_ylabel('Loss')
    axes.grid(True)
    axes.set_title('Evolution of the loss')
    plt.savefig(save_path, dpi=300, transparent=False, bbox_inches='tight')