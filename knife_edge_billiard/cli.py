"""Command-line entry point for the knife-edge billiard simulation."""
import argparse
import shutil
import sys

import matplotlib
matplotlib.use("Agg")  # headless, save-only
from matplotlib import pyplot as plt

from tqdm import tqdm, trange
from pydantic import ValidationError

from . import (
    Coin,
    InitialConditionError,
    SimConsistencyError,
    elastic_impact,
    free_space_transition,
    animate,
    visualize_snapshot,
    save_all_plots,
    make_run_dir,
    copy_config,
    first_impact,
)
from .config import SimConfig

FPS = 30  # interpolation / animation frame rate


def run(args):
    # 1. Create and verify config
    sim = SimConfig.from_yaml(args.config)
    billiard, coin = sim.to_dataclasses()

    # 2. Verify CLI args and setup
    num_impacts = int(args.num_impacts)
    if num_impacts <= 0:
        raise InitialConditionError("Number of impacts must be positive.")
    run_dir = make_run_dir()
    copy_config(args.config, run_dir)

    # 3. Print summary of configs
    print("=== Simulation Config ===")
    print(sim.pretty())
    print()

    # 4. Simulate
    print(f"[1/3] Simulating {args.num_impacts} impacts (saving snapshots)...")
    traj = []
    snapshots_dir = run_dir / "snapshots"
    diagnostics_dir = run_dir / "diagnostics"
    last_snapshot_path = None
    coin_before_impact = coin

    # Check if initial coin is already on the wall; if not, advance to first impact
    if not billiard.contains_coin_boundary(coin):
        coin_seq, free_space_data = first_impact(coin, billiard, interp_dt=1 / FPS)
        if len(coin_seq) == 0 or not isinstance(coin_seq[-1], Coin):
            raise SimConsistencyError(
                "First impact must return a non-empty list of Coin configurations."
            )
        if free_space_data is not None:
            traj.append({
                "impact_data": None,
                "free_space_data": free_space_data,
                "coin_seq": coin_seq,
            })
        coin_before_impact = coin_seq[-1]  # the last interpolated coin is the one exactly at impact

    for it in trange(num_impacts, desc="Simulating impacts"):
        # 1. Simulate an impact
        coin_after_impact, impact_data = elastic_impact(billiard, coin_before_impact)

        # 2. Simulate free-space dynamics until next impact
        coin_seq, free_space_data = free_space_transition(
            coin_after_impact, billiard, impact_data, 1 / FPS
        )

        # 3. Re-organize for storage
        traj.append({
            "impact_data": impact_data,
            "free_space_data": free_space_data,
            "coin_seq": coin_seq,
        })

        # 4. Save snapshot (headless)
        fig, _ = visualize_snapshot(billiard, traj)
        last_snapshot_path = snapshots_dir / f"snapshot_{it}.png"
        fig.savefig(last_snapshot_path)
        plt.close(fig)

        # 5. Update for next iteration
        if not isinstance(coin_seq, list) or len(coin_seq) == 0:
            raise SimConsistencyError(
                "Free-space transition must return a non-empty list of Coin configurations."
            )
        if not isinstance(coin_seq[-1], Coin):
            raise SimConsistencyError(
                "Free-space transition must return Coin configurations."
            )
        coin_before_impact = coin_seq[-1]

    # 5. Render video if selected
    if args.video:  
        print("[2/3] Rendering video (encoding with ffmpeg)...")
        total_video_frames = sum(len(seg["coin_seq"]) for seg in traj)
        with tqdm(total=total_video_frames, desc="Rendering video") as video_bar:
            def update_video_progress(current_frame, total_frames):
                if video_bar.total != total_frames:
                    video_bar.total = total_frames
                    video_bar.refresh()
                video_bar.update(current_frame + 1 - video_bar.n)

            animate(billiard, traj, save_path=run_dir / "trajectory.mp4", fps=FPS,
                    progress_callback=update_video_progress)
    else:
        print("[2/3] Skipping video rendering (use --video to enable)...")
    print()

    if last_snapshot_path is not None:
        shutil.copyfile(last_snapshot_path, run_dir / "trajectory.png")

    # 6. Save plots
    print("[3/3] Saving plots...")
    plot_traj = [seg for seg in traj if seg["impact_data"] is not None]
    save_all_plots(plot_traj, billiard, coin, diagnostics_dir)

    print(f"Run complete. Output saved to {run_dir}")
    return 0


def _format_validation_error(exc):
    """Render a pydantic ``ValidationError`` as a friendly, per-field summary."""
    errors = exc.errors()
    count = len(errors)
    noun = "problem" if count == 1 else "problems"
    lines = [f"The configuration is invalid ({count} {noun} found):"]
    for err in errors:
        loc = ".".join(str(part) for part in err["loc"]) or "(config root)"
        msg = err["msg"]
        # Pydantic prefixes custom ValueError messages with "Value error, ";
        # drop it so the message reads naturally.
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        lines.append(f"  • {loc}: {msg}")
    return "\n".join(lines)


def _friendly_error(exc):
    """Map an exception to a ``(message, exit_code)`` pair for the user.

    More specific types are checked first: both ``ValidationError`` and
    ``InitialConditionError`` are ``ValueError`` subclasses.
    """
    if isinstance(exc, ValidationError):
        return _format_validation_error(exc), 2
    if isinstance(exc, FileNotFoundError):
        return str(exc), 2
    if isinstance(exc, InitialConditionError):
        return f"Cannot start the simulation: {exc}", 2
    if isinstance(exc, SimConsistencyError):
        return (
            f"The simulation reached an inconsistent state: {exc}\n"
            "This usually points to a bug in the physics code rather than your config."
        ), 3
    if isinstance(exc, ValueError):
        return str(exc), 2
    return f"Unexpected error: {exc}", 1


def main():
    parser = argparse.ArgumentParser(
        description="Run the knife-edge billiard impact simulation from a config file."
    )
    parser.add_argument("-c", "--config", required=True,
                        help="Path to the config YAML (Billiard + Coin).")
    parser.add_argument("--num-impacts", type=int, default=20,
                        help="Number of impacts to simulate (default: 20).")
    parser.add_argument("--video", action="store_true",
                        help="Render an MP4 of the run (default: off).")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print the full traceback on error (for debugging).")
    args = parser.parse_args()

    try:
        return run(args)
    except Exception as exc:  # noqa: BLE001 - top-level friendly error boundary
        if args.verbose:
            raise
        message, code = _friendly_error(exc)
        print(f"Error: {message}", file=sys.stderr)
        print("\nRun again with -v/--verbose to see the full traceback.",
              file=sys.stderr)
        return code


if __name__ == "__main__":
    sys.exit(main())
