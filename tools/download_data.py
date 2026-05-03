#!/usr/bin/env python3
"""Download or resume the Embed2Heights dataset from EOTDL.

Requires authentication. Run `eotdl auth login` first if needed.

Usage:
    # Backward-compatible initial download
    python tools/download_data.py --path ../data

    # Explicit initial download
    python tools/download_data.py download --path ../data

    # Resume an interrupted asset download using the local catalog parquet
    python tools/download_data.py resume --path ../data --workers 8
    python tools/download_data.py resume --path ../data --dry-run
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def check_auth():
    """Verify that the user is authenticated with EOTDL."""
    try:
        from eotdl.auth import is_logged
    except ImportError:
        print("ERROR: eotdl package not installed.")
        print("Install it with:  pip install eotdl")
        print("Or create the conda environment:  conda env create -f environment.yml")
        sys.exit(1)

    try:
        if not is_logged():
            print("ERROR: Not authenticated with EOTDL.")
            print("Please run:  eotdl auth login")
            print("Then re-run this script.")
            sys.exit(1)
    except Exception:
        # Auth check may behave differently across versions;
        # let EOTDL's download calls raise a clear error if auth is missing.
        pass


def download_dataset(path):
    check_auth()

    from eotdl.datasets import stage_dataset

    print("Downloading embed2heights dataset to: {}".format(path))
    print("This may take a while depending on your connection speed...")
    print()

    try:
        stage_dataset("embed2heights", path=path, assets=True)
    except Exception as exc:
        print("\nERROR: Download failed: {}".format(exc))
        print()
        print("Common fixes:")
        print("  1. Check authentication:  eotdl auth login")
        print("  2. Check internet connection")
        print("  3. Check disk space (dataset is several GB)")
        sys.exit(1)

    print()
    print("Download complete! Dataset saved to: {}".format(path))
    print()
    print("Example training command:")
    print("  python train.py \\")
    print("      --train-embeddings-dir {}/embed2heights/<embedding_dir> \\".format(path))
    print("      --train-targets-dir {}/embed2heights/<labels_dir>".format(path))


def latest_catalog(dataset_dir):
    catalogs = sorted(Path(dataset_dir).glob("catalog.v*.parquet"))
    if not catalogs:
        raise FileNotFoundError("No catalog parquet found in {}".format(dataset_dir))
    return catalogs[-1]


def load_asset_urls(catalog_path):
    try:
        import geopandas as gpd
    except ImportError:
        print("ERROR: geopandas package not installed; required for resume mode.")
        print("Use the project environment or install geopandas.")
        sys.exit(1)

    gdf = gpd.read_parquet(catalog_path)
    urls = []
    for _, row in gdf.iterrows():
        for _, asset in row["assets"].items():
            urls.append(asset["href"])
    return urls


def staged_file_name(url):
    if "/stage/" in url:
        return url.split("/stage/")[-1]
    return url.split("//")[-1]


def find_missing_urls(dataset_dir, urls):
    missing = []
    for url in urls:
        file_path = os.path.join(dataset_dir, staged_file_name(url))
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            missing.append(url)
    return missing


def resume_dataset(path, workers=8, dry_run=False):
    check_auth()

    dataset_dir = os.path.join(path, "embed2heights")
    try:
        catalog_path = latest_catalog(dataset_dir)
    except FileNotFoundError as exc:
        print("ERROR: {}".format(exc))
        print("Run the initial download first: python tools/download_data.py --path {}".format(path))
        sys.exit(1)

    print("Using catalog: {}".format(catalog_path))
    all_urls = load_asset_urls(catalog_path)
    missing_urls = find_missing_urls(dataset_dir, all_urls)

    print("Total assets: {}".format(len(all_urls)))
    print("Already downloaded: {}".format(len(all_urls) - len(missing_urls)))
    print("Missing/empty: {}".format(len(missing_urls)))

    if not missing_urls:
        print("Nothing to download!")
        return

    if dry_run:
        for url in missing_urls[:20]:
            print("  {}".format(staged_file_name(url)))
        if len(missing_urls) > 20:
            print("  ... and {} more".format(len(missing_urls) - 20))
        return

    from eotdl.auth import with_auth
    from eotdl.repos import FilesAPIRepo

    @with_auth
    def get_user(user=None):
        return user

    user = get_user()
    repo = FilesAPIRepo()
    n_workers = min(int(workers), len(missing_urls))
    failed = []

    def download_one(href):
        try:
            repo.stage_file_url(href, dataset_dir, user)
        except Exception as exc:
            return href, str(exc)
        return None

    print("\nDownloading {} missing files ({} workers)...".format(len(missing_urls), n_workers))
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(download_one, href) for href in missing_urls]
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc="Downloading")
        for future in iterator:
            result = future.result()
            if result is not None:
                failed.append(result)

    if failed:
        print("\n{} files failed to download:".format(len(failed)))
        for href, err in failed[:10]:
            print("  {}: {}".format(staged_file_name(href), err))
        if len(failed) > 10:
            print("  ... and {} more".format(len(failed) - 10))
        print("\nRe-run resume mode to retry failed downloads.")
    else:
        print("\nAll {} missing files downloaded successfully!".format(len(missing_urls)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument(
        "--path",
        type=str,
        default="./data",
        help="Root output directory for the dataset (default: ./data)",
    )

    download = subparsers.add_parser("download", help="Initial EOTDL dataset download.")
    download.add_argument("--path", type=str, default="./data")

    resume = subparsers.add_parser("resume", help="Download missing/empty assets from an existing catalog.")
    resume.add_argument("--path", type=str, default="./data")
    resume.add_argument("--workers", type=int, default=8)
    resume.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    command = args.command or "download"

    if command == "download":
        download_dataset(args.path)
    elif command == "resume":
        resume_dataset(args.path, workers=args.workers, dry_run=args.dry_run)
    else:
        raise ValueError("Unknown command: {}".format(command))


if __name__ == "__main__":
    main()
