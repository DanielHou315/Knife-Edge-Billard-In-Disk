from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .data import Billiard, Coin

TrajectorySegment = Mapping[str, Any]


def _coin_sequence(segment: TrajectorySegment) -> list[Coin]:
    coin_seq = segment.get("coin_seq")
    if not isinstance(coin_seq, list) or not coin_seq:
        raise ValueError("Each trajectory segment must contain a non-empty 'coin_seq' list.")
    if not all(isinstance(coin, Coin) for coin in coin_seq):
        raise TypeError("Each item in 'coin_seq' must be a Coin.")
    return coin_seq


def _validated_trajectory(
    trajectory: Iterable[TrajectorySegment],
) -> list[TrajectorySegment]:
    trajectory = list(trajectory)
    if not trajectory:
        raise ValueError("Trajectory must contain at least one segment.")
    for segment in trajectory:
        _coin_sequence(segment)
    return trajectory


def _arc_colors(arc_count: int) -> list[Any]:
    return list(plt.colormaps["plasma"](np.linspace(0.0, 1.0, arc_count)))


def _draw_background(
    ax: Axes,
    billiard: Billiard,
    trajectory: list[TrajectorySegment],
    show_constructions: bool,
    show_samples: bool = True,
    show_arc_constructions: bool = True,
) -> None:
    ax.add_patch(Circle((0.0, 0.0), billiard.R, fill=False, color="black", lw=1.5))

    for segment in trajectory:
        coin_seq = _coin_sequence(segment)
        positions = np.array([coin.pos for coin in coin_seq])
        if show_samples:
            ax.scatter(positions[:, 0], positions[:, 1], color="black", s=1, alpha=0.65)
            ax.scatter(*positions[-1], color="red", s=10, zorder=4)

        # A straight (omega=0) segment has no finite arc centre (v/omega -> inf,
        # Sec 3.2), so skip the arc construction for it.
        if show_constructions and show_arc_constructions and coin_seq[0].omega != 0.0:
            arc_center = np.array([
                coin_seq[0].x - coin_seq[0].v / coin_seq[0].omega * np.sin(coin_seq[0].theta),
                coin_seq[0].y + coin_seq[0].v / coin_seq[0].omega * np.cos(coin_seq[0].theta),
            ])
            endpoints = positions[[0, -1]]
            ax.scatter(*arc_center, color="0.55", s=8, zorder=2)
            for endpoint in endpoints:
                ax.plot(
                    [arc_center[0], endpoint[0]],
                    [arc_center[1], endpoint[1]],
                    color="0.65",
                    ls=":",
                    lw=0.7,
                )

    margin = 0.08 * billiard.R
    ax.set(
        aspect="equal",
        xlim=(-billiard.R - margin, billiard.R + margin),
        ylim=(-billiard.R - margin, billiard.R + margin),
        xlabel="x",
        ylabel="y",
    )


def _coin_marker(ax: Axes, coin: Coin, color: Any) -> tuple[Line2D, Line2D, Line2D]:
    point, = ax.plot([coin.x], [coin.y], marker="o", color=color, ms=4.5, zorder=5)
    front, = ax.plot(
        [coin.x, coin.pos_front[0]],
        [coin.y, coin.pos_front[1]],
        color="red",
        lw=1.1,
        zorder=3,
    )
    back, = ax.plot(
        [coin.x, coin.pos_back[0]],
        [coin.y, coin.pos_back[1]],
        color="green",
        lw=1.1,
        zorder=3,
    )
    return point, front, back


def visualize_snapshot(
    billiard: Billiard,
    trajectory: Iterable[TrajectorySegment],
    *,
    show_constructions: bool = True,
) -> tuple[Figure, Axes]:
    """Plot time-discrete midpoint positions."""
    trajectory = _validated_trajectory(trajectory)

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    _draw_background(ax, billiard, trajectory, show_constructions)

    ax.set_title(f"Coin trajectory: {len(trajectory)} free-space arcs")
    return fig, ax


def animate(
    billiard: Billiard,
    trajectory: Iterable[TrajectorySegment],
    *,
    frame_step: int = 1,
    interval_ms: int = 1000 // 30,
    show_constructions: bool = True,
    repeat: bool = False,
    save_path: str | Path | None = None,
    fps: int = 30,
    progress_callback=None,
) -> FuncAnimation | None:
    """Animate the coin and optionally save the result as an MP4 file.

    ``progress_callback`` is forwarded to ``FuncAnimation.save`` and is called
    as ``progress_callback(current_frame, total_frames)`` during encoding.
    """
    trajectory = _validated_trajectory(trajectory)
    if frame_step <= 0:
        raise ValueError("frame_step must be positive.")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")

    colors = _arc_colors(len(trajectory))
    coins_with_arc_indices = [
        (coin, arc_index)
        for arc_index, segment in enumerate(trajectory)
        for coin in _coin_sequence(segment)
    ]

    if save_path is not None:
        save_path = Path(save_path)
        if save_path.suffix.lower() != ".mp4":
            raise ValueError("save_path must use the .mp4 extension.")

        coins = [coin for coin, _arc_index in coins_with_arc_indices]
        xs = np.array([coin.x for coin in coins], dtype=float)
        ys = np.array([coin.y for coin in coins], dtype=float)

        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        _draw_background(
            ax,
            billiard,
            trajectory,
            show_constructions,
            show_samples=False,
            show_arc_constructions=False,
        )
        trail, = ax.plot([], [], color="tab:blue", lw=1.2, alpha=0.9, zorder=3)
        point, front, back = _coin_marker(ax, coins[0], "tab:blue")

        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
        frame_indices = range(0, len(coins), frame_step)
        total_frames = len(range(0, len(coins), frame_step))
        with writer.saving(fig, str(save_path), dpi=120):
            for frame_number, index in enumerate(frame_indices):
                coin = coins[index]
                trail.set_data(xs[: index + 1], ys[: index + 1])
                point.set_data([coin.x], [coin.y])
                front.set_data([coin.x, coin.pos_front[0]], [coin.y, coin.pos_front[1]])
                back.set_data([coin.x, coin.pos_back[0]], [coin.y, coin.pos_back[1]])
                ax.set_title(f"Coin trajectory at t = {coin.t:.3f}")
                writer.grab_frame()
                if progress_callback is not None:
                    progress_callback(frame_number, total_frames)
        plt.close(fig)
        return None

    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    _draw_background(
        ax,
        billiard,
        trajectory,
        show_constructions,
        show_samples=False,
        show_arc_constructions=False,
    )

    trails = [
        ax.plot([], [], color=color, lw=1.2, alpha=0.9, zorder=3)[0]
        for color in colors
    ]
    construction_center, = ax.plot([], [], marker="o", color="0.55", ms=3, zorder=2)
    construction_start, = ax.plot([], [], color="0.65", ls=":", lw=0.7)
    construction_end, = ax.plot([], [], color="0.65", ls=":", lw=0.7)
    point, front, back = _coin_marker(ax, coins_with_arc_indices[0][0], colors[0])

    def update(index: int) -> tuple[Line2D, ...]:
        coin, arc_index = coins_with_arc_indices[index]
        color = colors[arc_index]
        for trail_index, trail in enumerate(trails):
            if trail_index > arc_index:
                trail.set_data([], [])
                continue
            arc_coins = _coin_sequence(trajectory[trail_index])
            if trail_index == arc_index:
                arc_coins = [
                    past_coin
                    for past_coin, past_arc_index in coins_with_arc_indices[: index + 1]
                    if past_arc_index == arc_index
                ]
            trail.set_data(
                [past_coin.x for past_coin in arc_coins],
                [past_coin.y for past_coin in arc_coins],
            )

        if show_constructions:
            arc_coins = _coin_sequence(trajectory[arc_index])
            arc_start = arc_coins[0]
            arc_end = arc_coins[-1]
            arc_center = np.array([
                arc_start.x - arc_start.v / arc_start.omega * np.sin(arc_start.theta),
                arc_start.y + arc_start.v / arc_start.omega * np.cos(arc_start.theta),
            ])
            construction_center.set_data([arc_center[0]], [arc_center[1]])
            construction_start.set_data(
                [arc_center[0], arc_start.x],
                [arc_center[1], arc_start.y],
            )
            construction_end.set_data(
                [arc_center[0], arc_end.x],
                [arc_center[1], arc_end.y],
            )
        else:
            construction_center.set_data([], [])
            construction_start.set_data([], [])
            construction_end.set_data([], [])

        point.set_data([coin.x], [coin.y])
        point.set_color(color)
        front.set_data(
            [coin.x, coin.pos_front[0]],
            [coin.y, coin.pos_front[1]],
        )
        back.set_data(
            [coin.x, coin.pos_back[0]],
            [coin.y, coin.pos_back[1]],
        )

        ax.set_title(f"Coin trajectory at t = {coin.t:.3f}")
        return (
            *trails,
            construction_center,
            construction_start,
            construction_end,
            point,
            front,
            back,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=range(0, len(coins_with_arc_indices), frame_step),
        interval=interval_ms,
        repeat=repeat,
        blit=False,
    )

    return animation


# --------------------------------------------------------------------------- #
# Per-quantity plots
# --------------------------------------------------------------------------- #


def _energy(coin: Coin) -> float:
    return 0.5 * coin.m * coin.v**2 + 0.5 * coin.J * coin.omega**2


def extract_series(traj, coin: Coin) -> dict[str, np.ndarray]:
    """Build per-impact arrays from the trajectory segments."""
    n = len(traj)
    series = {
        "impact": np.arange(n),
        "r": np.array([seg["free_space_data"].r_i for seg in traj], dtype=float),
        "d": np.array([seg["free_space_data"].d_i for seg in traj], dtype=float),
        "alpha": np.array([seg["free_space_data"].alpha_i for seg in traj], dtype=float),
        "beta": np.array([seg["free_space_data"].beta_i for seg in traj], dtype=float),
        "phi": np.array([seg["impact_data"].phi for seg in traj], dtype=float),
        "t": np.array([seg["impact_data"].t for seg in traj], dtype=float),
        "v": np.array([seg["coin_seq"][0].v for seg in traj], dtype=float),
        "omega": np.array([seg["coin_seq"][0].omega for seg in traj], dtype=float),
        "theta": np.array([seg["coin_seq"][0].theta for seg in traj], dtype=float),
    }
    series["energy_post"] = np.array([_energy(seg["coin_seq"][0]) for seg in traj], dtype=float)
    pre = np.empty(n, dtype=float)
    pre[0] = _energy(coin)
    for i in range(1, n):
        pre[i] = _energy(traj[i - 1]["coin_seq"][-1])
    series["energy_pre"] = pre
    series["dt"] = np.diff(series["t"])
    return series


def _save_line(out_dir, name, x, y, xlabel, ylabel, title, marker="o") -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(x, y, marker=marker, lw=1.2, ms=4)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(out_dir) / name)
    plt.close(fig)


def _save_energy(out_dir, s) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(s["impact"], s["energy_pre"], marker="o", lw=1.2, ms=4, label="pre-impact")
    ax.plot(s["impact"], s["energy_post"], marker="s", lw=1.2, ms=4, label="post-impact")
    ax.set(xlabel="impact #", ylabel="kinetic energy",
           title="Kinetic energy per impact (elastic ⇒ constant)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(Path(out_dir) / "energy_vs_impact.png")
    plt.close(fig)


def _save_phase(out_dir, s) -> None:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    ax.plot(s["v"], s["omega"], marker="o", lw=0.8, ms=4)
    ax.set(xlabel="v", ylabel="ω", title="(v, ω) phase portrait")
    ax.grid(True, alpha=0.3)
    fig.savefig(Path(out_dir) / "v_omega_phase.png")
    plt.close(fig)


def save_all_plots(traj, billiard: Billiard, coin: Coin, out_dir) -> dict[str, np.ndarray]:
    """Write one PNG per tracked quantity into out_dir. Returns the series dict."""
    out_dir = Path(out_dir)
    s = extract_series(traj, coin)

    _save_energy(out_dir, s)
    _save_line(out_dir, "arc_radius_vs_impact.png", s["impact"], s["r"],
               "impact #", "arc radius r", "Free-space arc radius per impact")
    _save_line(out_dir, "v_vs_impact.png", s["impact"], s["v"],
               "impact #", "v", "Forward velocity per impact")
    _save_line(out_dir, "omega_vs_impact.png", s["impact"], s["omega"],
               "impact #", "ω", "Angular velocity per impact")
    _save_phase(out_dir, s)
    _save_line(out_dir, "phi_vs_impact.png", s["impact"], s["phi"],
               "impact #", "φ", "Wall impact angle φ per impact")
    _save_line(out_dir, "theta_vs_impact.png", s["impact"], s["theta"],
               "impact #", "θ", "Coin orientation θ per impact")
    _save_line(out_dir, "alpha_vs_impact.png", s["impact"], s["alpha"],
               "impact #", "α", "Arc geometry angle α per impact")
    _save_line(out_dir, "beta_vs_impact.png", s["impact"], s["beta"],
               "impact #", "β", "Arc geometry angle β per impact")
    _save_line(out_dir, "d_vs_impact.png", s["impact"], s["d"],
               "impact #", "d", "Origin-to-arc-center distance per impact")
    _save_line(out_dir, "dt_between_impacts.png", s["impact"][1:], s["dt"],
               "impact #", "Δt", "Time between successive impacts")
    return s
