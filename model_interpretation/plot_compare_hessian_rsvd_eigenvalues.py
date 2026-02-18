import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def find_s_values_file(run_dir: Path) -> Optional[Path]:
	"""
	Find the *_S_rsvd.npy file inside a run directory.
	Returns the path if found, otherwise None.
	"""
	if not run_dir.exists() or not run_dir.is_dir():
		return None
	candidates = list(run_dir.glob("*_S_rsvd.npy"))
	if len(candidates) == 0:
		return None
	# Prefer exact single match; if multiple exist, take the first in sorted order
	candidates.sort()
	return candidates[0]


def load_eigenvalues_from_run(base_dir: Path, run_folder: str) -> Optional[np.ndarray]:
	"""
	Load eigenvalues (S values saved by RSVD flow) from the given run folder under base_dir.
	Returns an array of eigenvalues sorted descending, or None if missing.
	"""
	run_dir = base_dir / run_folder
	s_file = find_s_values_file(run_dir)
	if s_file is None:
		return None
	try:
		values = np.load(s_file)
	except Exception:
		return None
	# Ensure 1D float array, sorted descending for consistent spectrum plotting
	values = np.asarray(values, dtype=float).reshape(-1)
	values = np.sort(values)[::-1]
	return values


def ensure_output_dir(path: Path) -> None:
	path.mkdir(parents=True, exist_ok=True)


def build_default_series() -> List[Dict[str, str]]:
	"""
	Default configuration matching the user's request:
	- train/test regen_inception
	- train/test regen_inception_random_labels
	- train/test regen_inception_random_images
	- train_regen_inception_random_weights_seed_0 in black
	Train uses solid line, test uses dotted line.
	"""
	# return [
	# 	# Inception
	# 	{"folder": "train_regen_inception", "label": "Inception (train)", "color": "#1f77b4", "linestyle": "solid"},
	# 	{"folder": "test_regen_inception", "label": "Inception (test)", "color": "#1f77b4", "linestyle": "dotted"},
	# 	# Random labels
	# 	{"folder": "train_regen_inception_random_labels", "label": "Random labels (train)", "color": "#d62728", "linestyle": "solid"},
	# 	{"folder": "test_regen_inception_random_labels", "label": "Random labels (test)", "color": "#d62728", "linestyle": "dotted"},
	# 	# Random images
	# 	{"folder": "train_regen_inception_random_images", "label": "Random images (train)", "color": "#2ca02c", "linestyle": "solid"},
	# 	{"folder": "test_regen_inception_random_images", "label": "Random images (test)", "color": "#2ca02c", "linestyle": "dotted"},
	# 	# Random weights seed 0 in black
	# 	{"folder": "train_regen_inception_random_weights_seed_0", "label": "Random weights seed 0 (train)", "color": "#000000", "linestyle": "solid"},
	# ]

	# Grokking mod div
	return [
		{"folder": "train_grokking_modular_division_p97_epoch_73", "label": "Grokking mod div train (Bad Generalization)", "color": "#1f77b4", "linestyle": "solid"},
		{"folder": "test_grokking_modular_division_p97_epoch_73", "label": "Grokking mod div test (Bad Generalization)", "color": "#1f77b4", "linestyle": "dotted"},
		{"folder": "test_grokking_modular_division_p97", "label": "Grokking mod div test (Good Generalization)", "color": "#ff7f0e", "linestyle": "dotted"},
		{"folder": "train_grokking_modular_division_p97_last_epoch", "label": "Grokking mod div train (Good Generalization)", "color": "#ff7f0e", "linestyle": "solid"},
		{"folder": "train_grokking_modular_division_p97_random_weights_seed_0", "label": "Grokking mod div random weights seed 0", "color": "#000000", "linestyle": "solid"},
		{"folder": "train_grokking_modular_division_p97_random_weights_seed_1", "label": "Grokking mod div random weights seed 1", "color": "#000000", "linestyle": "solid"},
	]


def parse_series_config(config_path: Optional[Path]) -> Optional[List[Dict[str, str]]]:
	"""
	Parse a JSON config file describing the series to plot.
	The file should be a list of objects with keys:
	  - folder (str): subfolder under base_dir
	  - label (str): legend label
	  - color (str): matplotlib color
	  - linestyle (str): 'solid' or 'dotted'
	Returns None if config_path is None.
	"""
	if config_path is None:
		return None
	try:
		with config_path.open("r", encoding="utf-8") as f:
			series = json.load(f)
		if not isinstance(series, list):
			raise ValueError("Config JSON must be a list of series objects")
		for item in series:
			if not isinstance(item, dict):
				raise ValueError("Each series entry must be an object")
			for key in ("folder", "label", "color", "linestyle"):
				if key not in item:
					raise ValueError(f"Missing key '{key}' in series entry: {item}")
		return series
	except Exception as e:
		print(f"Failed to read config file '{config_path}': {e}", file=sys.stderr)
		return None


def linestyle_to_mpl(style: str) -> str:
	"""
	Map human strings to matplotlib linestyles.
	"""
	if style.lower() in ("solid", "s", "-"):
		return "-"
	if style.lower() in ("dotted", "dot", ":"):
		return ":"
	# Fallback to solid if unknown
	return "-"


def plot_series(
	base_dir: Path,
	series: List[Dict[str, str]],
	title: str,
	y_log: bool = True,
	x_label: str = "Eigenvalue Index",
	y_label: str = "Eigenvalue",
	x_lim: Optional[int] = None,
	y_min: Optional[float] = None,
	y_max: Optional[float] = None,
) -> List[str]:
	"""
	Plot the configured series. Returns a list of series folder names that were actually plotted.
	"""
	plotted_folders: List[str] = []
	plt.figure(figsize=(10, 6))

	for item in series:
		folder = item["folder"]
		label = item["label"]
		color = item["color"]
		linestyle = linestyle_to_mpl(item["linestyle"])

		values = load_eigenvalues_from_run(base_dir, folder)
		if values is None or values.size == 0:
			print(f"Warning: missing or empty eigenvalues for '{folder}', skipping.", file=sys.stderr)
			continue

		indices = np.arange(1, values.shape[0] + 1)
		if isinstance(x_lim, int) and x_lim > 0:
			limit = min(x_lim, values.shape[0])
			indices = indices[:limit]
			values = values[:limit]

		plt.plot(indices, values, linestyle=linestyle, color=color, linewidth=1.8, label=label)
		plotted_folders.append(folder)

	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	if y_log:
		plt.yscale("log")
	if (y_min is not None) or (y_max is not None):
		plt.ylim(bottom=y_min, top=y_max)
	plt.grid(True, alpha=0.3, which="both")
	plt.legend()
	plt.tight_layout()
	return plotted_folders


def build_output_filename(plotted_folders: List[str], prefix: str = "hessian_rsvd_spectrum") -> str:
	"""
	Construct an output filename that includes all the compared run folder names.
	Truncates or hashes when too long to avoid Windows MAX_PATH (260) errors.
	"""
	slug = "_".join(plotted_folders)
	max_slug_len = 120  # leave room for path + prefix + .png
	if len(slug) <= max_slug_len:
		return f"{prefix}_{slug}.png"
	import hashlib
	short = hashlib.md5(slug.encode()).hexdigest()[:10]
	return f"{prefix}_{len(plotted_folders)}_series_{short}.png"

def main():
	parser = argparse.ArgumentParser(description="Plot and compare Hessian RSVD eigenvalue spectra from saved outputs.")
	parser.add_argument(
		"--base-dir",
		type=str,
		default="model_interpretation/outputs/fisher_analysis_hessian",
		help="Base directory containing run subfolders with *_S_rsvd.npy files.",
	)
	parser.add_argument(
		"--config",
		type=str,
		default=None,
		help="Optional JSON config path overriding default series. Format: "
		     "[{\"folder\":\"...\",\"label\":\"...\",\"color\":\"#RRGGBB\",\"linestyle\":\"solid|dotted\"}, ...]",
	)
	parser.add_argument(
		"--title",
		type=str,
		default="Fisher Hessian (RSVD) Eigenvalue Spectrum Comparison",
		help="Plot title.",
	)
	parser.add_argument(
		"--no-logy",
		action="store_true",
		help="Disable log-scale for Y axis.",
	)
	parser.add_argument(
		"--top-k",
		type=int,
		default=None,
		help="Optionally plot only the top-K eigenvalues.",
	)
	parser.add_argument(
		"--ymin",
		type=float,
		default=None,
		help="Lower Y limit.",
	)
	parser.add_argument(
		"--ymax",
		type=float,
		default=None,
		help="Upper Y limit.",
	)
	parser.add_argument(
		"--output-name",
		type=str,
		default=None,
		help="Optional explicit output filename (placed under rsvd_comparison). If not provided, it is generated from plotted run names.",
	)
	args = parser.parse_args()

	base_dir = Path(args.base_dir)
	if not base_dir.exists():
		print(f"Base directory not found: {base_dir}", file=sys.stderr)
		sys.exit(1)

	series = parse_series_config(Path(args.config)) if args.config else None
	if series is None:
		series = build_default_series()

	plotted = plot_series(
		base_dir=base_dir,
		series=series,
		title=args.title,
		y_log=(not args.no_logy),
		x_lim=args.top_k,
		y_min=args.ymin,
		y_max=args.ymax,
	)

	if len(plotted) == 0:
		print("No series were plotted (missing files?). Nothing to save.", file=sys.stderr)
		sys.exit(2)

	# Save to rsvd_comparison with all run names in filename by default (project convention)
	output_dir = base_dir / "rsvd_comparison"
	ensure_output_dir(output_dir)
	filename = args.output_name if args.output_name else build_output_filename(plotted)
	out_path = output_dir / filename
	plt.savefig(out_path, dpi=200)
	print(f"Saved comparison figure to: {out_path}")


if __name__ == "__main__":
	main()



