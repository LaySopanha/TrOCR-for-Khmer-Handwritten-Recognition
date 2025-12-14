#!/usr/bin/env python3
"""
Rewrite Label Studio image URLs in an annotation export to be relative paths.

By default this script rewrites data/data/annotation/annotated_data.json so that
`data.image` values point to files in data/data/raw/ relative to the annotation
file location. Use the CLI flags to point at different inputs/outputs.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple


def update_image_paths(
    tasks: List[dict], image_dir: Path, base_dir: Path
) -> Tuple[int, List[str]]:
    """
    Update tasks in-place. Returns the number of paths updated and any that are missing.
    """
    updated = 0
    missing: List[str] = []

    for task in tasks:
        data_block = task.get("data")
        if not isinstance(data_block, dict) or "image" not in data_block:
            continue

        filename = Path(str(data_block["image"])).name
        if not filename:
            continue

        target = image_dir / filename
        relative_path = Path(os.path.relpath(target, start=base_dir)).as_posix()
        data_block["image"] = relative_path
        updated += 1

        if not target.exists():
            missing.append(relative_path)

    return updated, missing


def main() -> None:
    # Default paths are relative to the repo root (one level up from this script).
    repo_root = Path(__file__).resolve().parents[1]
    default_input = repo_root / "data/annotation/annotated_data.json"
    default_image_dir = repo_root / "data/image"

    parser = argparse.ArgumentParser(
        description="Rewrite Label Studio image URLs to local relative paths."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=default_input,
        help="Path to the Label Studio export JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the rewritten JSON (defaults to input file).",
    )
    parser.add_argument(
        "-d",
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Directory containing the image files.",
    )
    parser.add_argument(
        "-r",
        "--relative-to",
        type=Path,
        help="Directory the image paths should be relative to "
        "(defaults to output file's parent).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing the output file.",
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path
    image_dir = args.image_dir.resolve()
    base_dir = (args.relative_to or output_path.parent).resolve()

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    tasks = json.loads(input_path.read_text())
    updated, missing = update_image_paths(tasks, image_dir=image_dir, base_dir=base_dir)

    if args.dry_run:
        print(f"Would update {updated} annotation entries.")
        if missing:
            print(f"Warning: {len(missing)} image files not found:")
            for path in missing:
                print(f"  {path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tasks, indent=4))

    print(
        f"Wrote {output_path} with {updated} image paths "
        f"relative to {base_dir} using images from {image_dir}."
    )
    if missing:
        print(f"Warning: {len(missing)} image files not found:")
        for path in missing:
            print(f"  {path}")


if __name__ == "__main__":
    main()
