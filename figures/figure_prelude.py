#!/usr/bin/env python3
"""
Figure: Prelude to Figure 5
Generates animation frames showing how optimization campaigns produce the
data points summarized in Figure 5's box plots.

Left panel:  Individual campaign unfolding over 20 iterations
Right panel: Accumulating best observations from completed campaigns

Output: Individual PNG frames + GIF in figures/pngs/figure_prelude_frames/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_CAMPAIGNS = 20
N_ITERATIONS = 20
SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'pngs', 'figure_prelude_frames')
FIG_SIZE = (16, 9)
DPI = 200

# Colors
COLOR_OBS = '#BDBDBD'
COLOR_OBS_EDGE = '#616161'
COLOR_BEST_LINE = '#1565C0'
COLOR_BEST_FILL = '#1565C0'
COLOR_HIGHLIGHT = '#FFD600'
COLOR_HIGHLIGHT_EDGE = '#E65100'
COLOR_DOT = '#1565C0'
COLOR_NEW_DOT = '#FF6F00'

# Font
try:
    plt.rcParams['font.family'] = 'SF Pro Display'
except Exception:
    plt.rcParams['font.family'] = 'sans-serif'

# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------

def simulate_campaigns(n_campaigns, n_iterations, seed):
    """
    Generate simulated optimization campaigns with realistic behaviour.

    Each campaign has a different 'ceiling' -- the best value it can
    realistically discover.  Early iterations explore randomly; later
    iterations exploit better regions.  The result is a spread of
    final best-observations that resembles real optimisation data.
    """
    rng = np.random.RandomState(seed)

    # Draw ceilings from a beta distribution scaled to [55, 98]
    ceilings = np.sort(rng.beta(3.0, 1.3, n_campaigns)) * 43 + 55

    campaigns = []
    for ceiling in ceilings:
        obs = []
        for j in range(n_iterations):
            progress = (j + 1) / n_iterations
            # Exploration probability decreases over time
            if rng.random() < 0.35 - 0.2 * progress:
                val = rng.uniform(5, ceiling * 0.55)
            else:
                centre = ceiling * (0.30 + 0.60 * progress ** 0.6)
                spread = ceiling * 0.10 * (1 - 0.4 * progress)
                val = rng.normal(centre, spread)
            obs.append(np.clip(val, 0, 100))
        campaigns.append(np.array(obs))

    return campaigns

# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def make_frame(
    campaigns,
    current_campaign,
    current_iteration,
    completed_best_obs,
    highlight_best=False,
    show_transfer=False,
):
    """Render a single animation frame and return the Figure."""
    fig = plt.figure(figsize=FIG_SIZE, facecolor='white')
    gs = GridSpec(1, 2, width_ratios=[3, 1.1], wspace=0.08)

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # ------------------------------------------------------------------
    # Left panel -- current campaign
    # ------------------------------------------------------------------
    campaign = campaigns[current_campaign]
    iters = np.arange(1, current_iteration + 1)
    obs = campaign[:current_iteration]
    cum_best = np.maximum.accumulate(obs)

    # Light fill under cumulative best
    ax_left.fill_between(iters, 0, cum_best, color=COLOR_BEST_FILL,
                         alpha=0.07, step='post', zorder=1)

    # Cumulative best step line
    ax_left.step(iters, cum_best, where='post', color=COLOR_BEST_LINE,
                 linewidth=3, zorder=4, label='Cumulative best')

    # Raw observations
    ax_left.scatter(iters, obs, c=COLOR_OBS, edgecolors=COLOR_OBS_EDGE,
                    s=90, zorder=3, linewidths=1.5, label='Observations')

    # Highlight the iteration that achieved the best value
    if highlight_best and current_iteration == N_ITERATIONS:
        best_idx = int(np.argmax(obs))
        ax_left.scatter(
            [iters[best_idx]], [obs[best_idx]],
            c=COLOR_HIGHLIGHT, edgecolors=COLOR_HIGHLIGHT_EDGE,
            s=350, zorder=6, linewidths=3, marker='*',
        )
        # Horizontal dashed line at the best value
        ax_left.axhline(y=obs[best_idx], color=COLOR_HIGHLIGHT_EDGE,
                        linestyle='--', linewidth=1.5, alpha=0.6, zorder=2)

    # Axes formatting
    ax_left.set_xlim(0.5, N_ITERATIONS + 0.5)
    ax_left.set_xticks(range(1, N_ITERATIONS + 1))
    ax_left.set_ylim(-2, 105)
    ax_left.set_xlabel('Iteration', fontsize=20, fontweight='bold')
    ax_left.set_ylabel('Objective Value (%)', fontsize=20, fontweight='bold')
    ax_left.set_title(f'Campaign {current_campaign + 1} / {N_CAMPAIGNS}',
                      fontsize=24, fontweight='bold', pad=15)
    ax_left.tick_params(labelsize=15)
    ax_left.grid(axis='y', linestyle='--', alpha=0.4)
    ax_left.spines['top'].set_visible(False)
    ax_left.spines['right'].set_visible(False)
    for sp in ['left', 'bottom']:
        ax_left.spines[sp].set_linewidth(1.5)

    # ------------------------------------------------------------------
    # Right panel -- accumulating best observations
    # ------------------------------------------------------------------
    ax_right.set_ylim(-2, 105)
    ax_right.set_xlim(-0.8, 0.8)

    if completed_best_obs:
        n = len(completed_best_obs)
        # Deterministic jitter so dots don't jump between frames
        jitter_rng = np.random.RandomState(123)
        x_jitter = jitter_rng.uniform(-0.25, 0.25, n)

        if show_transfer and n > 0:
            # Previously-completed dots
            if n > 1:
                ax_right.scatter(
                    x_jitter[:n - 1], completed_best_obs[:n - 1],
                    c=COLOR_DOT, edgecolors='black', s=140,
                    zorder=3, linewidths=1.5, alpha=0.75,
                )
            # Newly-added dot (highlighted)
            ax_right.scatter(
                [x_jitter[n - 1]], [completed_best_obs[n - 1]],
                c=COLOR_NEW_DOT, edgecolors='black', s=220,
                zorder=4, linewidths=2,
            )
        else:
            ax_right.scatter(
                x_jitter[:n], completed_best_obs[:n],
                c=COLOR_DOT, edgecolors='black', s=140,
                zorder=3, linewidths=1.5, alpha=0.75,
            )

    # Counter label
    n_done = len(completed_best_obs)
    ax_right.text(0, -8, f'{n_done} / {N_CAMPAIGNS}',
                  ha='center', va='top', fontsize=15, color='#424242')

    ax_right.set_title('Best\nObservations', fontsize=20, fontweight='bold', pad=15)
    ax_right.set_xticks([])
    ax_right.tick_params(left=True, labelleft=True, labelsize=15)
    ax_right.yaxis.tick_right()
    ax_right.grid(axis='y', linestyle='--', alpha=0.4)
    ax_right.spines['top'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.spines['bottom'].set_visible(False)
    ax_right.spines['right'].set_linewidth(1.5)

    plt.tight_layout()
    return fig

# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------

def generate_frames():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    campaigns = simulate_campaigns(N_CAMPAIGNS, N_ITERATIONS, SEED)

    frame_paths = []
    frame_durations = []  # milliseconds, for GIF timing
    frame_num = 0
    completed_best_obs = []

    for c in range(N_CAMPAIGNS):
        campaign = campaigns[c]

        # --- Iteration frames ---
        for i in range(1, N_ITERATIONS + 1):
            fig = make_frame(campaigns, c, i, list(completed_best_obs))
            path = os.path.join(OUTPUT_DIR, f'frame_{frame_num:04d}.png')
            fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            frame_paths.append(path)
            frame_durations.append(120)
            frame_num += 1

        # --- Highlight frame (pause on completed campaign) ---
        fig = make_frame(campaigns, c, N_ITERATIONS, list(completed_best_obs),
                         highlight_best=True)
        path = os.path.join(OUTPUT_DIR, f'frame_{frame_num:04d}.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        frame_paths.append(path)
        frame_durations.append(700)
        frame_num += 1

        # --- Transfer frame (dot appears on right panel) ---
        best_val = float(np.max(campaign))
        completed_best_obs.append(best_val)

        fig = make_frame(campaigns, c, N_ITERATIONS, list(completed_best_obs),
                         highlight_best=True, show_transfer=True)
        path = os.path.join(OUTPUT_DIR, f'frame_{frame_num:04d}.png')
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        frame_paths.append(path)
        frame_durations.append(500)
        frame_num += 1

    # Hold last frame longer
    frame_durations[-1] = 2500

    print(f'Generated {frame_num} frames in {OUTPUT_DIR}/')

    # ------------------------------------------------------------------
    # Assemble GIF
    # ------------------------------------------------------------------
    try:
        from PIL import Image

        images = [Image.open(p) for p in frame_paths]
        gif_path = os.path.join(SCRIPT_DIR, 'pngs', 'figure_prelude.gif')
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=frame_durations,
            loop=0,
        )
        print(f'Generated GIF: {gif_path}')
    except ImportError:
        print('Pillow not installed -- skipping GIF generation.')

    return frame_paths, frame_durations


if __name__ == '__main__':
    generate_frames()
