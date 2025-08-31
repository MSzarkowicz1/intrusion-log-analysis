import matplotlib as mpl
import matplotlib.pyplot as plt

DARK_BG = "#0e1117"  # Streamlit dark background
FG = "#e5e7eb"  # light gray text/ticks
GRID = "#475569"  # muted grid


def apply_dark_theme() -> None:
    plt.style.use("dark_background")
    mpl.rcParams.update(
        {
            "figure.facecolor": DARK_BG,
            "axes.facecolor": DARK_BG,
            "axes.edgecolor": FG,
            "axes.labelcolor": FG,
            "xtick.color": FG,
            "ytick.color": FG,
            "grid.color": GRID,
            "text.color": FG,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def apply_light_theme() -> None:
    plt.style.use("default")
    mpl.rcParams.update(mpl.rcParamsDefault)
