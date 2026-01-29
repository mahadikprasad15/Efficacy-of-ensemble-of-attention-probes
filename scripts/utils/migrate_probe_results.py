#!/usr/bin/env python3
"""
Migration script to move plots from probe directories to results directories.

This script handles existing artifacts that were saved with the old structure
where plots were stored alongside probe files.

Usage:
    # Dry run (shows what would be moved)
    python scripts/utils/migrate_probe_results.py --dry-run

    # Actually move files
    python scripts/utils/migrate_probe_results.py

    # Also generate missing best_probe.json files
    python scripts/utils/migrate_probe_results.py --generate-best-probe

    # Specify custom directories (e.g., for Google Drive)
    python scripts/utils/migrate_probe_results.py \
        --probe_dirs /content/drive/MyDrive/probes_layer_agnostic \
        --results_base /content/drive/MyDrive/results
"""

import os
import sys
import argparse
import shutil
import json
import glob
from pathlib import Path

PLOT_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']


def find_plots_in_probe_dirs(probe_base_dirs):
    """Find all plot files that should be moved to results directories."""
    plots_to_move = []

    for probe_base in probe_base_dirs:
        if not os.path.exists(probe_base):
            print(f"  Skipping (not found): {probe_base}")
            continue

        for root, dirs, files in os.walk(probe_base):
            for file in files:
                if any(file.endswith(ext) for ext in PLOT_EXTENSIONS):
                    # Check if this looks like a probe directory (has .pt files)
                    pt_files = glob.glob(os.path.join(root, "*.pt"))
                    if pt_files:
                        plots_to_move.append({
                            'source': os.path.join(root, file),
                            'probe_dir': root,
                            'filename': file
                        })

    return plots_to_move


def compute_results_path(probe_dir, results_base_map):
    """Compute the corresponding results directory for a probe directory."""
    for probe_base, results_base in results_base_map.items():
        # Normalize paths for comparison
        probe_base_norm = os.path.normpath(probe_base)
        probe_dir_norm = os.path.normpath(probe_dir)

        if probe_dir_norm.startswith(probe_base_norm):
            relative = os.path.relpath(probe_dir_norm, probe_base_norm)
            return os.path.join(results_base, relative)
    return None


def find_json_results_to_move(probe_base_dirs):
    """Find JSON result files that should be moved."""
    results_to_move = []

    # Files that belong in results directory, not probe directory
    result_files = ['layer_results.json', 'results.json', 'combined_summary.json']

    for probe_base in probe_base_dirs:
        if not os.path.exists(probe_base):
            continue

        for root, dirs, files in os.walk(probe_base):
            # Check if this is a probe directory
            pt_files = glob.glob(os.path.join(root, "*.pt"))
            if not pt_files:
                continue

            for file in files:
                if file in result_files:
                    results_to_move.append({
                        'source': os.path.join(root, file),
                        'probe_dir': root,
                        'filename': file
                    })

    return results_to_move


def migrate_files(files_to_move, results_base_map, dry_run=False):
    """Move files from probe directories to results directories."""
    moved = []
    errors = []

    for file_info in files_to_move:
        probe_dir = file_info['probe_dir']
        source = file_info['source']
        filename = file_info['filename']

        results_dir = compute_results_path(probe_dir, results_base_map)
        if not results_dir:
            errors.append(f"Could not determine results dir for: {source}")
            continue

        dest = os.path.join(results_dir, filename)

        if dry_run:
            print(f"  Would move: {source}")
            print(f"         To: {dest}")
        else:
            try:
                os.makedirs(results_dir, exist_ok=True)
                shutil.move(source, dest)
                print(f"  Moved: {source} -> {dest}")
                moved.append({'source': source, 'dest': dest})
            except Exception as e:
                errors.append(f"Error moving {source}: {e}")

    return moved, errors


def generate_missing_best_probe_json(probe_base_dirs, dry_run=False):
    """Generate best_probe.json for directories that don't have it."""
    generated = []

    for probe_base in probe_base_dirs:
        if not os.path.exists(probe_base):
            continue

        for root, dirs, files in os.walk(probe_base):
            # Check if this is a probe directory (has .pt files but no best_probe.json)
            pt_files = glob.glob(os.path.join(root, "*.pt"))
            best_probe_path = os.path.join(root, "best_probe.json")

            if pt_files and not os.path.exists(best_probe_path):
                # Try to find results file (could be in same dir or results dir)
                results_file = None
                for name in ['layer_results.json', 'results.json']:
                    candidate = os.path.join(root, name)
                    if os.path.exists(candidate):
                        results_file = candidate
                        break

                if not results_file:
                    # Results might have already been moved
                    continue

                print(f"  Found probe dir without best_probe.json: {root}")

                if not dry_run:
                    try:
                        with open(results_file, 'r') as f:
                            results = json.load(f)

                        # Handle different result formats
                        if isinstance(results, list):
                            # Standard layer_results.json format
                            best = max(results, key=lambda x: x.get('val_auc', x.get('auc', 0)))
                            best_layer = best.get('layer', 0)
                            best_auc = best.get('val_auc', best.get('auc', 0))
                        elif isinstance(results, dict) and 'id_per_layer' in results:
                            # Layer-agnostic format
                            id_results = results['id_per_layer']
                            # Keys might be strings
                            best_layer = max(id_results.keys(), key=lambda k: id_results[k]['auc'])
                            best_auc = id_results[best_layer]['auc']
                            best_layer = int(best_layer)
                        elif isinstance(results, dict) and 'mean' in results:
                            # Combined per-token format (aggregation -> layer results)
                            # Use mean aggregation by default
                            mean_results = results.get('mean', [])
                            if mean_results:
                                best = max(mean_results, key=lambda x: (x.get('auc_a', 0) + x.get('auc_b', 0)) / 2)
                                best_layer = best.get('layer', 0)
                                best_auc = (best.get('auc_a', 0) + best.get('auc_b', 0)) / 2
                            else:
                                continue
                        else:
                            print(f"    Unknown results format in {results_file}, skipping")
                            continue

                        # Determine probe path
                        if os.path.exists(os.path.join(root, 'probe.pt')):
                            probe_path = os.path.join(root, 'probe.pt')
                            probe_type = 'layer_agnostic'
                        else:
                            probe_path = os.path.join(root, f'probe_layer_{best_layer}.pt')
                            probe_type = 'standard'

                        # Infer pooling from directory name
                        pooling = os.path.basename(root)
                        if pooling not in ['mean', 'max', 'last', 'attn', 'none']:
                            pooling = 'unknown'

                        best_probe_info = {
                            'probe_type': probe_type,
                            'best_layer': int(best_layer),
                            'best_val_auc': float(best_auc),
                            'probe_path': probe_path,
                            'pooling': pooling,
                            'migrated': True,
                            'note': 'Auto-generated by migration script'
                        }

                        with open(best_probe_path, 'w') as f:
                            json.dump(best_probe_info, f, indent=2)

                        print(f"    Generated: {best_probe_path}")
                        generated.append(best_probe_path)

                    except Exception as e:
                        print(f"    Error: {e}")

    return generated


def main():
    parser = argparse.ArgumentParser(
        description="Migrate plots and results from probe dirs to results dirs"
    )
    parser.add_argument('--probe_dirs', nargs='+', default=[
        'data/probes_layer_agnostic',
        'data/probes_combined_per_token'
    ], help='Probe directories to process')
    parser.add_argument('--results_base', type=str, default='results',
                        help='Base directory for results (default: results)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    parser.add_argument('--generate-best-probe', action='store_true',
                        help='Generate missing best_probe.json files')
    parser.add_argument('--move-json', action='store_true',
                        help='Also move JSON result files (layer_results.json, etc.)')
    args = parser.parse_args()

    # Build mapping from probe base dirs to results base dirs
    results_base_map = {}
    for probe_dir in args.probe_dirs:
        # Extract the probe type from directory name (e.g., probes_layer_agnostic -> probes_layer_agnostic)
        probe_type = os.path.basename(probe_dir)
        results_base_map[probe_dir] = os.path.join(args.results_base, probe_type)

    print("=" * 70)
    print("PROBE RESULTS MIGRATION")
    print("=" * 70)

    if args.dry_run:
        print("DRY RUN - No changes will be made\n")

    print(f"Probe directories: {args.probe_dirs}")
    print(f"Results mapping:")
    for p, r in results_base_map.items():
        print(f"  {p} -> {r}")
    print()

    # Find plots to move
    print("1. Finding plots in probe directories...")
    plots = find_plots_in_probe_dirs(args.probe_dirs)
    print(f"   Found {len(plots)} plot(s) to migrate\n")

    # Migrate plots
    if plots:
        print("2. Migrating plots...")
        moved, errors = migrate_files(plots, results_base_map, args.dry_run)
        print(f"   Migrated: {len(moved)}, Errors: {len(errors)}")
        if errors:
            for err in errors:
                print(f"   ERROR: {err}")
        print()

    # Move JSON files if requested
    if args.move_json:
        print("3. Finding JSON result files...")
        json_files = find_json_results_to_move(args.probe_dirs)
        print(f"   Found {len(json_files)} JSON file(s) to migrate\n")

        if json_files:
            print("4. Migrating JSON files...")
            moved, errors = migrate_files(json_files, results_base_map, args.dry_run)
            print(f"   Migrated: {len(moved)}, Errors: {len(errors)}")
            if errors:
                for err in errors:
                    print(f"   ERROR: {err}")
            print()

    # Generate best_probe.json if requested
    if args.generate_best_probe:
        step = 5 if args.move_json else 3
        print(f"{step}. Generating missing best_probe.json files...")
        generated = generate_missing_best_probe_json(args.probe_dirs, args.dry_run)
        print(f"   Generated: {len(generated)} file(s)\n")

    print("=" * 70)
    print("MIGRATION COMPLETE" if not args.dry_run else "DRY RUN COMPLETE")
    print("=" * 70)

    # Print summary of new structure
    if not args.dry_run:
        print("\nNew structure:")
        print("  Probes remain in: data/probes_*/...")
        print("  Results moved to: results/probes_*/...")
        print("\nRun with --dry-run first to preview changes.")


if __name__ == "__main__":
    main()
