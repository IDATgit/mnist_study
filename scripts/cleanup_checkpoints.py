import argparse
import os
import re
import sys
from typing import List, Tuple


CHECKPOINT_DIR_NAME = "checkpoints"
CHECKPOINT_REGEX = re.compile(r"^checkpoint_epoch_(\d+)\.pt$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Delete checkpoint files whose epoch index is not a multiple of N (default: 10)."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=r"D:\Projects\Deep_Learning\mnist_test_platform\trainers\outputs",
        help="Root outputs directory containing model folders.",
    )
    parser.add_argument(
        "--multiple-of",
        type=int,
        default=10,
        help="Keep only checkpoints where epoch %% N == 0.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without deleting.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed actions.",
    )
    return parser.parse_args()


def find_checkpoint_dirs(root: str) -> List[str]:
    checkpoint_dirs: List[str] = []
    if not os.path.isdir(root):
        return checkpoint_dirs
    # Only look for immediate children model folders and their checkpoints subdir
    try:
        for entry in os.scandir(root):
            if not entry.is_dir():
                continue
            candidate = os.path.join(entry.path, CHECKPOINT_DIR_NAME)
            if os.path.isdir(candidate):
                checkpoint_dirs.append(candidate)
    except PermissionError:
        pass
    return checkpoint_dirs


def classify_checkpoints(checkpoint_dir: str, keep_mod: int) -> Tuple[List[str], List[str]]:
    keep: List[str] = []
    delete: List[str] = []
    try:
        for entry in os.scandir(checkpoint_dir):
            if not entry.is_file():
                continue
            filename = entry.name
            match = CHECKPOINT_REGEX.match(filename)
            if not match:
                # Ignore non-matching files
                continue
            epoch_str = match.group(1)
            try:
                epoch = int(epoch_str)
            except ValueError:
                # Ignore unparsable epoch files
                continue
            if epoch % keep_mod == 0:
                keep.append(entry.path)
            else:
                delete.append(entry.path)
    except PermissionError:
        pass
    return keep, delete


def human_rel(path: str, root: str) -> str:
    try:
        return os.path.relpath(path, root)
    except ValueError:
        # On Windows, relpath can raise if on different drives
        return path


def main() -> int:
    args = parse_args()
    root = args.root
    keep_mod = args.multiple_of

    if not os.path.isdir(root):
        print(f"Root directory not found: {root}", file=sys.stderr)
        return 1

    checkpoint_dirs = find_checkpoint_dirs(root)
    if args.verbose:
        print(f"Found {len(checkpoint_dirs)} checkpoint directories under {root}")
        for d in checkpoint_dirs:
            print(f" - {human_rel(d, root)}")

    total_keep = 0
    total_delete = 0
    total_errors = 0

    for ckpt_dir in checkpoint_dirs:
        model_dir = os.path.dirname(ckpt_dir)
        model_name = os.path.basename(model_dir)
        keep, delete = classify_checkpoints(ckpt_dir, keep_mod)
        total_keep += len(keep)
        total_delete += len(delete)

        if args.verbose:
            print(f"\nModel: {model_name}")
            print(f"  Checkpoints dir: {human_rel(ckpt_dir, root)}")
            print(f"  Keep ({len(keep)}):")
            for p in sorted(keep):
                print(f"    {os.path.basename(p)}")
            print(f"  Delete ({len(delete)}):")
            for p in sorted(delete):
                print(f"    {os.path.basename(p)}")

        if not args.dry_run and delete:
            for path in delete:
                try:
                    os.remove(path)
                except Exception as exc:  # noqa: BLE001
                    total_errors += 1
                    if args.verbose:
                        print(f"    ERROR deleting {path}: {exc}", file=sys.stderr)

    print(
        f"\nSummary (multiple of {keep_mod}): keep={total_keep}, "
        f"delete={'planned' if args.dry_run else 'deleted'}={total_delete}, "
        f"errors={total_errors}"
    )
    if args.dry_run:
        print("Dry-run mode; no files were deleted. Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


