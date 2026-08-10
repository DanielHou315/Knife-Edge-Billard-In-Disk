# Knife-Edge Billiard in a Disk

Simulation code for `The Integrability of a knife-edge Billiard in a Disk`.

## Installation (macOS)

1. Install system tools
```
# install uv for python management
curl -LsSf https://astral.sh/uv/install.sh | sh

# If not installed, uncomment and install homebrew:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# install ffmpeg
brew install ffmpeg
```

`ffmpeg` is only required when using `--video`.

2. Download code and install Python dependencies
```
git clone https://github.com/DanielHou315/Knife-Edge-Billard-In-Disk.git
cd Knife-Edge-Billard-In-Disk
uv sync
```

## Configuration

The config contains **only** the `billiard` and `coin` parameters:

```yaml
billiard:
  R: 0.2

coin:
  x: 0.0
  y: 0.0
  theta: "pi/2"
  v: 0.1
  omega: 0.2
  l: 0.01
  m: 0.0025
```

`theta` may be a number or a safe expression string using `pi`, numeric
literals, parentheses, and `+`, `-`, `*`, `/`, for example `"pi/2"`,
`"-pi/4"`, or `"2*pi"`. It is normalized to `[-pi, pi)`.

`l` is the **half**-length: the coin spans `[-l, +l]` about its midpoint, so
its full length is `2l`.

### Moment of inertia

`J` is not an independent parameter — it is fixed by the mass and the
geometry, so it is normally omitted and derived as the uniform bar

```
J = m * l^2 / 3        (equivalently m * ell^2 / 12 for the full length ell = 2l)
```

The symmetry requirement (centre of mass at the midpoint) does not pin the
mass distribution down uniquely, so `J` may still be given explicitly to model
a different symmetric body. It is then bounded by the limiting case of all the
mass sitting at the two endpoints:

```
J <= m * l^2
```

Configurations violating that bound correspond to no rigid body and produce
nonsensical trajectories; they are rejected.

The config is validated before running:

1. every `billiard`/`coin` field except `J` must be present
2. unknown config keys are rejected
3. `R`, `l`, `m`, and — when given — `J` must be positive
4. an explicit `J` must satisfy `J <= m * l^2`
5. the coin must lie entirely inside or touching the billiard

Reference config files are provided in `configs/`

## Run the Example

`uv sync` installs the project itself, so the `knife-edge-billiard` command is
available inside the project environment.

Run the simulation from a YAML config:
```bash
uv run knife-edge-billiard -c configs/penny-coin.yaml
```

Equivalently, as a module:
```bash
uv run python -m knife_edge_billiard -c configs/penny-coin.yaml
```

Required flags:
- `-c/--config`: the config YAML file path defining the initial conditions and simulation settings of the coin. 

Optional flags:

- `--num-impacts N`: number of impacts to simulate, default `20`. If the
  initial state is already at impact, that impact is counted as the first one.
- `--video`: render `trajectory.mp4`. Video rendering uses `ffmpeg` and a `tqdm` progress bar. If `ffmpeg` is not installed, naive rendering speed will be noticeably slower.
- `-v/--verbose`: on error, print the full traceback instead of a short, friendly message (config validation, initial-condition, and simulation errors).

Each run writes a timestamped folder under `data/`:

```text
data/2026-06-02_14-57-03/
├── config.yaml
├── trajectory.mp4         # only when --video is used
├── trajectory.png         # copy of the final snapshot
├── diagnostics/
│   ├── energy_vs_impact.png
│   └── ...                # other per-quantity plots
└── snapshots/
    └── snapshot_0.png ...
```

Runs are headless and save-only; no windows pop up.

## Tests

Run the deterministic randomized regression tests with:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl-cache \
uv run python -m unittest discover -s tests -v
```