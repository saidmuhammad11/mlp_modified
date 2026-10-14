#!/usr/bin/env python3
"""
MLP-RPM energy/performance comparison and publication-ready plots.

Updated to support multi-run averaging across baseline and proposed experimental results.
All secondary figures follow the clearer journal template.
Secondary plots remove CPU frequency and temperature to focus on primary performance metrics.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from collections import Counter

# =================== CONFIGURATION ===================
# Provide lists of JSON log files for multiple experiment runs.
BASELINE_FILES = [
    './results/memory_intensive_medium_run1/monitoring.jsonl',
    './results/memory_intensive_medium_run2/monitoring.jsonl',
    './results/memory_intensive_medium_run3/monitoring.jsonl',
    './results/memory_intensive_medium_run4/monitoring.jsonl',
    './results/memory_intensive_medium_run5/monitoring.jsonl',
]

PROPOSED_FILES = [
    './results/memory_intensive_mlp_run1/monitoring.jsonl',
    './results/memory_intensive_mlp_run2/monitoring.jsonl',
    './results/memory_intensive_mlp_run3/monitoring.jsonl',
    './results/memory_intensive_mlp_run4/monitoring.jsonl',
    './results/memory_intensive_mlp_run5/monitoring.jsonl',
]


# Every run gets its own timestamped results folder so previous experiments are preserved.
RESULTS_ROOT = 'experimental_results'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
RESULTS_DIR = os.path.join(RESULTS_ROOT, f'mlp_rpm_{RUN_ID}')

# Figure filename stems. Each figure is saved as BOTH JPEG and PDF.
OUTPUT_PLOT_COMPREHENSIVE = 'mlp_rpm_comprehensive_analysis'
OUTPUT_PLOT_BOX = 'mlp_rpm_box_plots'
OUTPUT_PLOT_LEVELS = 'mlp_rpm_performance_levels'
OUTPUT_PLOT_CACHE = 'mlp_rpm_cache_performance'
OUTPUT_TABLE_FIGURE = 'mlp_rpm_comparison_table'

# Non-image outputs are saved in the same experiment folder.
OUTPUT_TABLE_CSV = os.path.join(RESULTS_DIR, 'mlp_rpm_comparison_results.csv')
OUTPUT_REPORT = os.path.join(RESULTS_DIR, 'mlp_rpm_analysis_report.txt')

SAMPLING_INTERVAL_SEC = 3
PLOT_DURATION_SEC = 300.0  # Fallback duration (5 min)

# Analysis time grid settings
ANALYSIS_DURATION_S = 300.0  # Dynamically updated in main based on max run duration
COMMON_GRID_STEP_S = 1.0
STRICT_DURATION_CHECK = False  # Allows runs with slightly varying completion times
ENDPOINT_TOLERANCE_MULTIPLIER = 1.5
MIN_ENDPOINT_TOLERANCE_S = 3.0
WORKLOAD_NAME = "CPU_Intensive"

# Figure/report labels used throughout the script.
BASELINE_LABEL = "Default Windows Medium"
PROPOSED_LABEL = "MLP-RPM"

# Final printed width used by the aligned manuscript template.
FULL_WIDTH_IN = 7.16
BASE_FONT_SIZE = 10.5
PANEL_LABEL_SIZE = BASE_FONT_SIZE
AUX_TITLE_SIZE = 11.2
AUX_LABEL_SIZE = 11.2
AUX_TICK_SIZE = 10.2
AUX_LEGEND_SIZE = 10.5

# Journal-template typography for secondary figures.
LEVEL_LABEL_SIZE = 11.2
LEVEL_TICK_SIZE = 10.2
LEVEL_TITLE_SIZE = 11.2
LEVEL_COUNT_SIZE = 10.2
LEVEL_PANEL_SIZE = 11.2

SUMMARY_LABEL_SIZE = 11.8
SUMMARY_TICK_SIZE = 10.8
SUMMARY_LEGEND_SIZE = 11.8
SUMMARY_PANEL_SIZE = 11.8

# Colors
COLOR_WITH_DVFS = '#FF69B4'      # Pink: MLP-RPM
COLOR_default_DVFS = '#4169E1'   # Blue: Default Windows Medium

STATE_STYLES = {
    'Low': {'color': '#4C78A8', 'hatch': '///'},
    'Medium': {'color': '#F2A541', 'hatch': 'xx'},
    'High': {'color': '#59A14F', 'hatch': '...'},
    'Fixed': {'color': '#9D9D9D', 'hatch': '\\'},
}

CACHE_STYLES = {
    'L2': {'color': '#59A14F'},
    'L3': {'color': '#E15759'},
}

POLICY_STYLES = {
    BASELINE_LABEL: {'hatch': '///'},
    PROPOSED_LABEL: {'hatch': 'xx'},
}

# Memory bandwidth threshold
MAX_MEMORY_BANDWIDTH = 50.0  # GB/s

# =================== PUBLICATION FIGURE STYLE ===================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': BASE_FONT_SIZE,
    'axes.titlesize': BASE_FONT_SIZE,
    'axes.labelsize': BASE_FONT_SIZE,
    'xtick.labelsize': BASE_FONT_SIZE,
    'ytick.labelsize': BASE_FONT_SIZE,
    'legend.fontsize': BASE_FONT_SIZE,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.0,
    'grid.linewidth': 0.45,
    'grid.alpha': 0.25,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
    'savefig.dpi': 300,
    'figure.dpi': 120,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})


def ensure_results_dir():
    """Create the output directory for this experimental run."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def save_figure_jpeg_pdf(fig, filename_stem, *, bbox_inches='tight', pad_inches=None):
    """Save one Matplotlib figure as both a 300-dpi JPEG and a vector PDF."""
    ensure_results_dir()
    jpeg_path = os.path.join(RESULTS_DIR, f'{filename_stem}.jpg')
    pdf_path = os.path.join(RESULTS_DIR, f'{filename_stem}.pdf')

    save_kwargs = {'bbox_inches': bbox_inches}
    if pad_inches is not None:
        save_kwargs['pad_inches'] = pad_inches

    fig.savefig(jpeg_path, dpi=300, **save_kwargs)
    fig.savefig(pdf_path, **save_kwargs)
    return jpeg_path, pdf_path


def add_panel_label(ax, label):
    """Place a publication-style panel label at the upper-left."""
    ax.text(
        -0.12, 1.03, label, transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE, fontweight='bold',
        va='bottom', ha='left'
    )


def align_ylabels_by_column(fig, axes, left_x=-0.17, right_x=-0.16):
    """Align y-axis labels vertically within each subplot column."""
    axes = np.asarray(axes, dtype=object)
    for row in range(axes.shape[0]):
        for col in range(axes.shape[1]):
            ax = axes[row, col]
            ax.yaxis.set_label_coords(left_x if col == 0 else right_x, 0.5)
    for col in range(axes.shape[1]):
        fig.align_ylabels([axes[row, col] for row in range(axes.shape[0])])


def load_json_log(filename):
    """Load and parse JSON log file into DataFrame with elapsed seconds."""
    if not os.path.exists(filename):
        print(f"Warning: File not found, skipping: {filename}")
        return pd.DataFrame()
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entry = json.loads(line)
                    if 'dvfs_level' not in entry:
                        entry['dvfs_level'] = "Fixed"
                    data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    if not data:
        print(f"Warning: No valid data found in {filename}")
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['elapsed_s'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    return df


def trapezoid(y, x):
    """Trapezoidal integration with compatibility across NumPy versions."""
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_energy(df, power_col, fallback_dt=SAMPLING_INTERVAL_SEC):
    """Compute total energy in Joules using recorded timestamps when available."""
    if df.empty or power_col not in df.columns:
        return np.nan

    valid = df[['elapsed_s', power_col]].dropna().copy()
    if len(valid) < 2:
        return float(valid[power_col].sum()) * fallback_dt

    t = valid['elapsed_s'].to_numpy(dtype=float)
    p = valid[power_col].to_numpy(dtype=float)
    return float(trapezoid(p, t))


def estimate_sampling_interval_s(df):
    """Estimate median positive telemetry interval in seconds."""
    elapsed = df['elapsed_s'].to_numpy(dtype=float)
    differences = np.diff(elapsed)
    positive = differences[np.isfinite(differences) & (differences > 0.0)]
    if positive.size == 0:
        return float('nan')
    return float(np.median(positive))


def endpoint_tolerance_s(df):
    """Maximum small endpoint gap permitted for run duration."""
    median_interval = estimate_sampling_interval_s(df)
    if not np.isfinite(median_interval):
        return MIN_ENDPOINT_TOLERANCE_S
    return max(
        MIN_ENDPOINT_TOLERANCE_S,
        ENDPOINT_TOLERANCE_MULTIPLIER * median_interval,
    )


def build_common_time_grid(duration_s):
    """Return the fixed uniform journal-analysis grid in seconds."""
    points = int(round(duration_s / COMMON_GRID_STEP_S))
    return np.linspace(0.0, duration_s, points + 1)


def numeric_metric_samples(df, column):
    """Return finite, timestamp-sorted numeric samples for one metric."""
    if column not in df.columns:
        raise KeyError(f"Missing metric column: {column}")

    clean = pd.DataFrame({
        'elapsed_s': df['elapsed_s'],
        'value': pd.to_numeric(df[column], errors='coerce'),
    }).dropna()
    clean = clean[np.isfinite(clean['elapsed_s']) & np.isfinite(clean['value'])]
    clean = clean.sort_values('elapsed_s').drop_duplicates('elapsed_s', keep='last')

    if len(clean) < 2:
        raise ValueError(f"Metric '{column}' has fewer than two valid samples")

    return (
        clean['elapsed_s'].to_numpy(dtype=float),
        clean['value'].to_numpy(dtype=float),
    )


def interpolate_numeric_metric(df, column, time_grid_s, label):
    """Interpolate a metric onto the common 0..T s analysis grid."""
    sample_time_s, sample_values = numeric_metric_samples(df, column)
    tolerance = endpoint_tolerance_s(df)
    first_time = float(sample_time_s[0])

    if STRICT_DURATION_CHECK and first_time > tolerance:
        raise ValueError(
            f"{label} metric '{column}' begins at {first_time:.2f} s; "
            f"permitted start gap is {tolerance:.2f} s."
        )

    aligned = np.interp(time_grid_s, sample_time_s, sample_values)
    if not np.all(np.isfinite(aligned)):
        raise ValueError(f"Aligned metric '{column}' contains non-finite values")
    return aligned.astype(float, copy=False)


def align_categorical_state(df, time_grid_s):
    """Align controller states by last-observation carry-forward."""
    if 'dvfs_level' not in df.columns:
        return np.full(time_grid_s.shape, 'Fixed', dtype=object)

    clean = pd.DataFrame({
        'elapsed_s': df['elapsed_s'],
        'dvfs_level': df['dvfs_level'],
    }).dropna()
    if clean.empty:
        return np.full(time_grid_s.shape, 'Fixed', dtype=object)

    clean = clean.sort_values('elapsed_s').drop_duplicates('elapsed_s', keep='last')
    sample_time_s = clean['elapsed_s'].to_numpy(dtype=float)
    sample_states = clean['dvfs_level'].astype(str).to_numpy(dtype=object)

    indices = np.searchsorted(sample_time_s, time_grid_s, side='right') - 1
    indices = np.clip(indices, 0, len(sample_states) - 1)
    return sample_states[indices]


def summarize_aligned_metric(time_s, baseline_values, proposed_values):
    """Return time-weighted means and baseline-referenced difference."""
    duration_s = float(time_s[-1] - time_s[0])
    if duration_s <= 0.0:
        raise ValueError('Aligned analysis duration must be positive')

    delta_values = baseline_values - proposed_values
    baseline_mean = trapezoid(baseline_values, time_s) / duration_s
    proposed_mean = trapezoid(proposed_values, time_s) / duration_s
    mean_delta = trapezoid(delta_values, time_s) / duration_s

    return {
        'baseline_mean': float(baseline_mean),
        'proposed_mean': float(proposed_mean),
        'mean_delta': float(mean_delta),
        'percentage_change': (
            100.0 * (proposed_mean - baseline_mean) / baseline_mean
            if baseline_mean != 0.0 else float('nan')
        ),
    }


class RobustDVFSComparator:
    """Multi-run comparator and figure generator for MLP-RPM."""

    def __init__(self, baselines, proposeds, workload_name="Unknown"):
        self.baselines = baselines
        self.proposeds = proposeds
        self.workload_name = workload_name
        self.color_baseline = COLOR_default_DVFS
        self.color_dvfs = COLOR_WITH_DVFS

        self.analysis_time_s = build_common_time_grid(ANALYSIS_DURATION_S)
        self.analysis_time_min = self.analysis_time_s / 60.0
        self._aligned_cache = {}
        self._stats_cache = {}

        self.workload_type = self.detect_workload_type()

    def detect_workload_type(self):
        """Detect if workload is CPU or memory intensive across runs."""
        avg_l3_miss = (
            self.metric_statistics('l3_miss_rate')['proposed_mean']
            if 'l3_miss_rate' in self.proposeds[0].columns else 0.0
        )
        avg_mem_bw = (
            self.metric_statistics('memory_bandwidth')['proposed_mean']
            if 'memory_bandwidth' in self.proposeds[0].columns else 0.0
        )

        if avg_l3_miss > 0.05 and avg_mem_bw > 0.6 * MAX_MEMORY_BANDWIDTH:
            return "memory_intensive"
        return "cpu_intensive"

    def aligned_metric(self, column):
        """Return cached baseline/proposed arrays averaged on the common grid."""
        if column not in self._aligned_cache:
            base_arrays = [
                interpolate_numeric_metric(df, column, self.analysis_time_s, f"{BASELINE_LABEL} Run {i+1}")
                for i, df in enumerate(self.baselines)
            ]
            prop_arrays = [
                interpolate_numeric_metric(df, column, self.analysis_time_s, f"{PROPOSED_LABEL} Run {i+1}")
                for i, df in enumerate(self.proposeds)
            ]
            self._aligned_cache[column] = (
                np.mean(base_arrays, axis=0),
                np.mean(prop_arrays, axis=0)
            )
        return self._aligned_cache[column]

    def metric_statistics(self, column):
        """Return cached time-weighted statistics for an aligned metric."""
        if column not in self._stats_cache:
            baseline_values, proposed_values = self.aligned_metric(column)
            self._stats_cache[column] = summarize_aligned_metric(
                self.analysis_time_s, baseline_values, proposed_values
            )
        return self._stats_cache[column]

    def create_comprehensive_plots(self):
        """Create the manuscript-style 3x2 runtime comparison figure (averaged traces)."""
        # Modified to 3 rows x 2 columns (6 subplots total)
        fig, axes = plt.subplots(
            3, 2,
            figsize=(FULL_WIDTH_IN, 5.2),
            sharex='col',
            squeeze=False,
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.115, right=0.99, bottom=0.12, top=0.88,
            hspace=0.34, wspace=0.40
        )

        # Removed cpu_frequency and cpu_temperature
        metrics = [
            ('cpu_power', 'CPU power (W)'),
            ('ipc', 'IPC'),
            ('memory_usage', 'Memory usage (%)'),
            ('l2_miss_rate', 'L2 miss rate'),
            ('l3_miss_rate', 'L3 miss rate'),
            ('cpu_usage_overall', 'CPU usage (%)'),
        ]

        legend_handles = None
        legend_labels = None

        for index, (col, ylabel) in enumerate(metrics):
            row, column = divmod(index, 2)
            ax = axes[row, column]

            if col in self.baselines[0].columns and col in self.proposeds[0].columns:
                baseline_aligned, proposed_aligned = self.aligned_metric(col)

                baseline_line, = ax.plot(
                    self.analysis_time_min, baseline_aligned,
                    label=BASELINE_LABEL, color=self.color_baseline,
                    linewidth=1.05, alpha=0.95
                )
                proposed_line, = ax.plot(
                    self.analysis_time_min, proposed_aligned,
                    label=PROPOSED_LABEL, color=self.color_dvfs,
                    linewidth=1.05, alpha=0.95
                )
                if legend_handles is None:
                    legend_handles = [baseline_line, proposed_line]
                    legend_labels = [BASELINE_LABEL, PROPOSED_LABEL]

            ax.set_ylabel(ylabel, fontsize=BASE_FONT_SIZE, labelpad=2.0)
            ax.tick_params(
                axis='both', which='major',
                labelsize=BASE_FONT_SIZE, pad=1.5
            )
            ax.grid(True)
            add_panel_label(ax, f"({chr(ord('a') + index)})")

            if row < 2:  # Suppress bottom x ticks for top 2 rows
                ax.tick_params(axis='x', which='both', labelbottom=False)
            else:
                ax.set_xlabel('Time (min)', fontsize=BASE_FONT_SIZE, labelpad=2.0)

        plot_end = float(self.analysis_time_min[-1])
        minute_ticks = np.arange(0.0, plot_end + 0.001, 1.0)
        for ax in axes.ravel():
            ax.margins(x=0)
            ax.set_xmargin(0)
            ax.set_xlim(left=0.0, right=plot_end)
            ax.set_xticks(minute_ticks)
            ax.autoscale(enable=False, axis='x')

        align_ylabels_by_column(fig, axes)

        if legend_handles is not None:
            fig.legend(
                legend_handles, legend_labels,
                loc='upper center', ncol=2, frameon=False,
                bbox_to_anchor=(0.5, 0.975),
                borderaxespad=0.0, columnspacing=1.15,
                handlelength=1.8, handletextpad=0.5,
                fontsize=BASE_FONT_SIZE,
                prop={'size': BASE_FONT_SIZE, 'weight': 'medium'},
            )

        jpeg_path, pdf_path = save_figure_jpeg_pdf(
            fig, OUTPUT_PLOT_COMPREHENSIVE, bbox_inches='tight', pad_inches=0.03
        )
        plt.show()
        plt.close(fig)
        print(f"✅ Comprehensive plot saved:\n   JPEG: {jpeg_path}\n   PDF:  {pdf_path}")

    def create_box_plots(self):
        """Create the journal-template 2x2 distribution comparison from averaged arrays."""
        requested = [
            ('cpu_power', 'CPU power (W)'),
            ('ipc', 'IPC'),
            ('l2_miss_rate', 'L2 miss rate'),
            ('l3_miss_rate', 'L3 miss rate'),
        ]
        metrics = [
            pair for pair in requested
            if pair[0] in self.baselines[0].columns and pair[0] in self.proposeds[0].columns
        ]
        if not metrics:
            print('⚠️ No metrics available for box plots.')
            return

        fig, axes = plt.subplots(
            2, 2,
            figsize=(FULL_WIDTH_IN, 4.6),
            squeeze=False,
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.10, right=0.99, bottom=0.15, top=0.96,
            hspace=0.42, wspace=0.34,
        )

        for index, (column, ylabel) in enumerate(metrics):
            row, col = divmod(index, 2)
            ax = axes[row, col]
            baseline_values, proposed_values = self.aligned_metric(column)

            box = ax.boxplot(
                [baseline_values, proposed_values],
                tick_labels=[BASELINE_LABEL, PROPOSED_LABEL],
                patch_artist=True,
                widths=0.55,
            )
            for patch, color in zip(
                box['boxes'], [self.color_baseline, self.color_dvfs]
            ):
                patch.set_facecolor(color)
                patch.set_alpha(0.45)

            ax.set_ylabel(ylabel, labelpad=2.0)
            ax.tick_params(axis='x', rotation=12, pad=2.0)
            ax.tick_params(axis='y', pad=1.5)
            ax.grid(True, axis='y')
            add_panel_label(ax, f"({chr(ord('a') + index)})")

        for index in range(len(metrics), 4):
            row, col = divmod(index, 2)
            axes[row, col].set_visible(False)

        align_ylabels_by_column(fig, axes, left_x=-0.14, right_x=-0.13)
        jpeg_path, pdf_path = save_figure_jpeg_pdf(fig, OUTPUT_PLOT_BOX)
        plt.show()
        plt.close(fig)
        print(f"✅ Journal-style box plots saved:\n   JPEG: {jpeg_path}\n   PDF:  {pdf_path}")

    def create_performance_level_plots(self):
        """Plot journal-style analysis-time share in each state concatenated across runs."""
        baseline_states = np.concatenate([
            align_categorical_state(b, self.analysis_time_s) for b in self.baselines
        ])
        proposed_states = np.concatenate([
            align_categorical_state(p, self.analysis_time_s) for p in self.proposeds
        ])

        fig, axes = plt.subplots(
            1, 2,
            figsize=(FULL_WIDTH_IN, 3.85),
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.10, right=0.99, bottom=0.20, top=0.88, wspace=0.34
        )

        self._plot_distribution(
            axes[0], baseline_states, BASELINE_LABEL, self.color_baseline, '(a)'
        )
        self._plot_distribution(
            axes[1], proposed_states, PROPOSED_LABEL, self.color_dvfs, '(b)'
        )

        align_ylabels_by_column(
            fig, np.asarray(axes).reshape(1, -1),
            left_x=-0.13, right_x=-0.13,
        )

        jpeg_path, pdf_path = save_figure_jpeg_pdf(fig, OUTPUT_PLOT_LEVELS)
        plt.show()
        plt.close(fig)
        print(
            f"✅ Journal-style performance-level plot saved:\n"
            f"   JPEG: {jpeg_path}\n   PDF:  {pdf_path}"
        )

    @staticmethod
    def _plot_distribution(ax, states, title, fallback_color, panel_label):
        """Plot percentage of analysis time spent in each controller state."""
        counts = Counter(states.tolist())
        preferred_order = ['Low', 'Medium', 'High', 'Fixed']
        categories = sorted(
            counts,
            key=lambda value: (
                preferred_order.index(value)
                if value in preferred_order else len(preferred_order)
            ),
        )
        total = max(len(states), 1)
        percentages = [100.0 * counts[category] / total for category in categories]
        bar_colors = [
            STATE_STYLES.get(category, {'color': fallback_color})['color']
            for category in categories
        ]

        bars = ax.bar(
            categories, percentages,
            color=bar_colors,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.7,
        )

        for bar, category, percentage in zip(bars, categories, percentages):
            bar.set_hatch(STATE_STYLES.get(category, {}).get('hatch', ''))
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                percentage,
                f'{percentage:.1f}%',
                ha='center', va='bottom',
                fontsize=LEVEL_COUNT_SIZE,
            )

        ax.set_ylabel(
            'Analysis-time share (%)', fontsize=LEVEL_LABEL_SIZE, labelpad=4.0
        )
        ax.set_title(title, fontsize=LEVEL_TITLE_SIZE, pad=6)
        ax.tick_params(
            axis='both', which='major', labelsize=LEVEL_TICK_SIZE, pad=2.0
        )
        ax.grid(True, axis='y')
        ax.text(
            -0.11, 1.04, panel_label,
            transform=ax.transAxes,
            fontsize=LEVEL_PANEL_SIZE,
            fontweight='bold',
            va='bottom', ha='left',
        )

    def create_cache_performance_plot(self):
        """Create the journal-template mean L2/L3 hit- and miss-rate summary."""
        from matplotlib.patches import Patch

        required = {'l2_miss_rate', 'l3_miss_rate'}
        if not required.issubset(self.baselines[0].columns) or not required.issubset(
            self.proposeds[0].columns
        ):
            print('⚠️ L2/L3 miss-rate columns are unavailable; skipping cache plot.')
            return

        cache_levels = ['L2', 'L3']
        baseline_miss = [
            self.metric_statistics('l2_miss_rate')['baseline_mean'],
            self.metric_statistics('l3_miss_rate')['baseline_mean'],
        ]
        proposed_miss = [
            self.metric_statistics('l2_miss_rate')['proposed_mean'],
            self.metric_statistics('l3_miss_rate')['proposed_mean'],
        ]
        baseline_hit = [1.0 - value for value in baseline_miss]
        proposed_hit = [1.0 - value for value in proposed_miss]

        fig, axes = plt.subplots(
            1, 2,
            figsize=(FULL_WIDTH_IN, 3.45),
            constrained_layout=False,
        )
        fig.subplots_adjust(
            left=0.09, right=0.99, bottom=0.20, top=0.68, wspace=0.38
        )

        x = np.arange(len(cache_levels), dtype=float)
        width = 0.34
        panel_data = [
            (axes[0], baseline_hit, proposed_hit, 'Mean hit rate', '(a)'),
            (axes[1], baseline_miss, proposed_miss, 'Mean miss rate', '(b)'),
        ]

        for ax, baseline_values, proposed_values, ylabel, panel_label in panel_data:
            baseline_bars = ax.bar(
                x - width / 2,
                baseline_values,
                width,
                color=[CACHE_STYLES[level]['color'] for level in cache_levels],
                edgecolor='black', linewidth=0.85, alpha=0.72,
            )
            proposed_bars = ax.bar(
                x + width / 2,
                proposed_values,
                width,
                color=[CACHE_STYLES[level]['color'] for level in cache_levels],
                edgecolor='black', linewidth=0.85, alpha=1.0,
            )

            for bar in baseline_bars:
                bar.set_hatch(POLICY_STYLES[BASELINE_LABEL]['hatch'])
            for bar in proposed_bars:
                bar.set_hatch(POLICY_STYLES[PROPOSED_LABEL]['hatch'])

            ax.set_xticks(x)
            ax.set_xticklabels(cache_levels)
            ax.set_ylabel(ylabel, fontsize=SUMMARY_LABEL_SIZE, labelpad=4.0)
            ax.tick_params(
                axis='both', which='major', labelsize=SUMMARY_TICK_SIZE, pad=2.0
            )
            ax.grid(True, axis='y')
            ax.text(
                -0.12, 1.03, panel_label,
                transform=ax.transAxes,
                fontsize=SUMMARY_PANEL_SIZE,
                fontweight='bold',
                va='bottom', ha='left',
            )

        axes[0].set_ylim(0.0, 1.0)
        miss_upper = max(max(baseline_miss), max(proposed_miss))
        axes[1].set_ylim(0.0, max(0.10, miss_upper * 1.20))
        axes[0].yaxis.set_label_coords(-0.15, 0.5)
        axes[1].yaxis.set_label_coords(-0.15, 0.5)

        cache_handles = [
            Patch(facecolor=CACHE_STYLES['L2']['color'], edgecolor='black', label='L2'),
            Patch(facecolor=CACHE_STYLES['L3']['color'], edgecolor='black', label='L3'),
        ]
        policy_handles = [
            Patch(
                facecolor='white', edgecolor='black',
                hatch=POLICY_STYLES[BASELINE_LABEL]['hatch'], label=BASELINE_LABEL,
            ),
            Patch(
                facecolor='white', edgecolor='black',
                hatch=POLICY_STYLES[PROPOSED_LABEL]['hatch'], label=PROPOSED_LABEL,
            ),
        ]

        cache_legend = fig.legend(
            handles=cache_handles,
            loc='upper center', ncol=2, frameon=False,
            bbox_to_anchor=(0.28, 0.985),
            fontsize=SUMMARY_LEGEND_SIZE,
            title='Cache level', title_fontsize=SUMMARY_LEGEND_SIZE,
            handlelength=2.0, handletextpad=0.45, columnspacing=1.1,
        )
        fig.add_artist(cache_legend)
        fig.legend(
            handles=policy_handles,
            loc='upper center', ncol=2, frameon=False,
            bbox_to_anchor=(0.73, 0.985),
            fontsize=SUMMARY_LEGEND_SIZE,
            title='Policy', title_fontsize=SUMMARY_LEGEND_SIZE,
            handlelength=2.0, handletextpad=0.45, columnspacing=1.1,
        )

        jpeg_path, pdf_path = save_figure_jpeg_pdf(fig, OUTPUT_PLOT_CACHE)
        plt.show()
        plt.close(fig)
        print(
            f"✅ Journal-style cache hit/miss summary saved:\n"
            f"   JPEG: {jpeg_path}\n   PDF:  {pdf_path}"
        )

    def create_comparison_table(self):
        """Generate detailed comparison table with multi-run averaged metrics."""
        E_baseline = float(np.mean([compute_energy(b, 'cpu_power') for b in self.baselines]))
        E_dvfs = float(np.mean([compute_energy(p, 'cpu_power') for p in self.proposeds]))

        metrics = [
            ('Avg CPU Power (W)', self.metric_statistics('cpu_power')['baseline_mean'], self.metric_statistics('cpu_power')['proposed_mean'], 'lower'),
            ('Avg IPC', self.metric_statistics('ipc')['baseline_mean'], self.metric_statistics('ipc')['proposed_mean'], 'higher'),
            ('Avg CPU Usage (%)', self.metric_statistics('cpu_usage_overall')['baseline_mean'], self.metric_statistics('cpu_usage_overall')['proposed_mean'], 'lower'),
            ('Avg L2 Miss Rate', self.metric_statistics('l2_miss_rate')['baseline_mean'], self.metric_statistics('l2_miss_rate')['proposed_mean'], 'lower'),
            ('Avg L3 Miss Rate', self.metric_statistics('l3_miss_rate')['baseline_mean'], self.metric_statistics('l3_miss_rate')['proposed_mean'], 'lower'),
            ('Total Energy (J)', E_baseline, E_dvfs, 'lower'),
        ]

        data = {
            'Metric': [],
            BASELINE_LABEL: [],
            PROPOSED_LABEL: [],
            'Improvement': [],
            'Improvement %': []
        }

        for name, base_val, dvfs_val, direction in metrics:
            imp = dvfs_val - base_val
            imp_pct = (abs(imp) / base_val) * 100 if base_val != 0 else 0.0
            sign = '+' if (imp < 0 and direction == 'lower') or (imp > 0 and direction == 'higher') else '-'
            imp_str = f"{imp:+.3f}"
            imp_pct_str = f"{sign}{imp_pct:.1f}%"

            data['Metric'].append(name)
            data[BASELINE_LABEL].append(f"{base_val:.3f}")
            data[PROPOSED_LABEL].append(f"{dvfs_val:.3f}")
            data['Improvement'].append(imp_str)
            data['Improvement %'].append(imp_pct_str)

        df_table = pd.DataFrame(data)
        ensure_results_dir()
        df_table.to_csv(OUTPUT_TABLE_CSV, index=False)

        # Plot table
        fig, ax = plt.subplots(figsize=(14, len(df_table)*0.9))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=df_table.values,
                        colLabels=df_table.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(AUX_TICK_SIZE)
        table.scale(1.5, 2.5)

        for i in range(len(df_table)):
            pct = df_table.iloc[i]['Improvement %']
            is_positive = (pct.startswith('+') and float(pct.strip('%+')) > 0)
            metric = df_table.iloc[i]['Metric']
            if 'Energy' in metric and is_positive:
                table[(i+1, 4)].set_facecolor('#90EE90')
            elif not is_positive:
                table[(i+1, 4)].set_facecolor('#FFB6C1')

        plt.title('MLP-RPM Performance Comparison (Averaged)', fontsize=AUX_TITLE_SIZE, fontweight='bold', pad=10)
        jpeg_path, pdf_path = save_figure_jpeg_pdf(fig, OUTPUT_TABLE_FIGURE)
        plt.show()
        plt.close(fig)
        print(
            f"📋 Comparison table saved:\n"
            f"   CSV:  {OUTPUT_TABLE_CSV}\n"
            f"   JPEG: {jpeg_path}\n"
            f"   PDF:  {pdf_path}"
        )

    def generate_report(self):
        """Generate a detailed text report with averaged workload-aware insight."""
        E_baseline = float(np.mean([compute_energy(b, 'cpu_power') for b in self.baselines]))
        E_dvfs = float(np.mean([compute_energy(p, 'cpu_power') for p in self.proposeds]))
        energy_savings = ((E_baseline - E_dvfs) / E_baseline) * 100 if E_baseline else 0.0

        ipc_stats = self.metric_statistics('ipc')
        ipc_change = ipc_stats['percentage_change']

        l3_stats = self.metric_statistics('l3_miss_rate')
        l3_miss_change = ((l3_stats['baseline_mean'] - l3_stats['proposed_mean']) / l3_stats['baseline_mean']) * 100 if l3_stats['baseline_mean'] else 0.0

        avg_duration = float(np.mean([b['elapsed_s'].iloc[-1] for b in self.baselines]))

        report = f"""
MLP-RPM Performance Analysis Report (Multi-Run Averaged)
=========================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Workload: {self.workload_name}
Type: {self.workload_type.upper()}
Average Duration: {avg_duration:.1f} seconds
Processed Runs: {len(self.baselines)} baseline runs, {len(self.proposeds)} MLP-RPM runs

WORKLOAD CHARACTERIZATION
-------------------------
This is a {self.workload_type.replace('_', '-')} workload:
- High L3 miss rate and memory bandwidth → CPU stalls on memory
- Frequency scaling has limited impact on IPC
- Ideal strategy: Moderate frequency to save energy without hurting performance

EXECUTIVE SUMMARY
-----------------
- Energy Savings: {energy_savings:+.1f}% {'✅' if energy_savings > 0 else '❌'}
- IPC Change: {ipc_change:+.1f}% {'✅' if ipc_change > 0 else '❌'}
- L3 Miss Rate Reduction: {l3_miss_change:+.1f}% {'✅' if l3_miss_change > 0 else '❌'}

KEY METRICS (Averaged across runs)
----------------------------------
{BASELINE_LABEL}:
  Avg Power:     {self.metric_statistics('cpu_power')['baseline_mean']:.2f} W
  Avg IPC:       {self.metric_statistics('ipc')['baseline_mean']:.3f}
  L3 Miss Rate:  {self.metric_statistics('l3_miss_rate')['baseline_mean']:.3f}

{PROPOSED_LABEL}:
  Avg Power:     {self.metric_statistics('cpu_power')['proposed_mean']:.2f} W
  Avg IPC:       {self.metric_statistics('ipc')['proposed_mean']:.3f}
  L3 Miss Rate:  {self.metric_statistics('l3_miss_rate')['proposed_mean']:.3f}

MLP-RPM Level Distribution (Concatenated):
"""
        proposed_states = np.concatenate([
            align_categorical_state(p, self.analysis_time_s) for p in self.proposeds
        ])
        state_counts = Counter(proposed_states.tolist())
        total_samples = len(proposed_states)
        for level, count in state_counts.items():
            report += f"  {level}: {count} samples ({(count/total_samples)*100:.1f}%)\n"

        report += f"""
CONCLUSION
----------
"""
        if self.workload_type == "memory_intensive":
            if energy_savings > 5 and abs(ipc_change) < 3:
                report += "✅ The MLP-RPM controller is effective: it reduces energy significantly with negligible performance loss — ideal for memory-bound apps."
            elif energy_savings > 0 and ipc_change > -5:
                report += "🟡 The MLP-RPM controller saves energy with acceptable IPC trade-off."
            else:
                report += "❌ The MLP-RPM controller is too aggressive — it boosts frequency unnecessarily, wasting power without improving performance."
        else:
            if energy_savings > 5 and ipc_change > -2:
                report += "✅ The MLP-RPM controller is highly effective: significant energy savings with minimal performance impact."
            elif energy_savings > 0 and ipc_change > -5:
                report += "🟡 The MLP-RPM controller reduces energy with acceptable performance trade-off."
            else:
                report += "❌ The MLP-RPM controller does not provide clear benefits under this workload."

        ensure_results_dir()
        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write(report)
        print(report)
        print(f"📝 Full report saved to: {OUTPUT_REPORT}")


def main():
    print("🚀 Starting MLP-RPM Comparison Analysis (Multi-Run Average)")
    print("=" * 60)

    try:
        ensure_results_dir()
        print(f"📁 Results directory: {os.path.abspath(RESULTS_DIR)}")

        baselines = [load_json_log(f) for f in BASELINE_FILES]
        proposeds = [load_json_log(f) for f in PROPOSED_FILES]

        baselines = [b for b in baselines if not b.empty]
        proposeds = [p for p in proposeds if not p.empty]

        if not baselines or not proposeds:
            raise ValueError("Missing valid baseline or proposed log files. Check file paths.")

        max_b_dur = max(float(b['elapsed_s'].iloc[-1]) for b in baselines)
        max_p_dur = max(float(p['elapsed_s'].iloc[-1]) for p in proposeds)

        global ANALYSIS_DURATION_S
        ANALYSIS_DURATION_S = max(max_b_dur, max_p_dur)

        print(f"Loaded {len(baselines)} baseline runs and {len(proposeds)} MLP-RPM runs.")
        print(f"Auto-configured interpolation duration to {ANALYSIS_DURATION_S:.2f} seconds.")

        comparator = RobustDVFSComparator(baselines, proposeds, WORKLOAD_NAME)

        print("\n1. Generating comprehensive time-series plots...")
        comparator.create_comprehensive_plots()

        print("\n2. Generating box plots...")
        comparator.create_box_plots()

        print("\n3. Generating performance level plots...")
        comparator.create_performance_level_plots()

        print("\n4. Generating cache performance plot...")
        comparator.create_cache_performance_plot()

        print("\n5. Creating comparison table...")
        comparator.create_comparison_table()

        print("\n6. Generating analysis report...")
        comparator.generate_report()

        print("\n🎉 All multi-run averaged analysis complete!")
        print(f"📁 All experimental results saved in: {os.path.abspath(RESULTS_DIR)}")
        print("🖼️  Every figure/table image was saved as both .jpg and .pdf")
        print(f"📋 CSV: {OUTPUT_TABLE_CSV}")
        print(f"📄 Report: {OUTPUT_REPORT}")

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()