#!/usr/bin/env python3
"""Resume an interrupted embed2heights download.

Reads the catalog parquet to discover all expected asset files,
checks which ones already exist on disk, and downloads only the missing ones.

Usage:
    python tools/resume_download.py
    python tools/resume_download.py --path ./data --workers 4
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import geopandas as gpd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--path", type=str, default="./data",
                        help="Root download directory (default: ./data)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download workers (default: 8)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only report missing files, don't download")
    args = parser.parse_args()

    dataset_dir = os.path.join(args.path, "embed2heights")

    # Find the catalog
    catalogs = sorted(Path(dataset_dir).glob("catalog.v*.parquet"))
    if not catalogs:
        print(f"ERROR: No catalog parquet found in {dataset_dir}")
        print("Run the initial download first: python tools/download_data.py")
        sys.exit(1)

    catalog_path = str(catalogs[-1])
    print(f"Using catalog: {catalog_path}")

    gdf = gpd.read_parquet(catalog_path)
    all_urls = []
    for _, row in gdf.iterrows():
        for _, v in row["assets"].items():
            all_urls.append(v["href"])

    # Check which files already exist
    missing_urls = []
    for url in all_urls:
        # Replicate the path logic from stage_file_url
        if "/stage/" in url:
            file_name = url.split("/stage/")[-1]
        else:
            file_name = url.split("//")[-1]
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            missing_urls.append(url)

    print(f"Total assets: {len(all_urls)}")
    print(f"Already downloaded: {len(all_urls) - len(missing_urls)}")
    print(f"Missing/empty: {len(missing_urls)}")

    if not missing_urls:
        print("Nothing to download!")
        return

    if args.dry_run:
        for url in missing_urls[:20]:
            file_name = url.split("/stage/")[-1] if "/stage/" in url else url.split("//")[-1]
            print(f"  {file_name}")
        if len(missing_urls) > 20:
            print(f"  ... and {len(missing_urls) - 20} more")
        return

    # Download missing files
    from eotdl.auth import with_auth
    from eotdl.repos import FilesAPIRepo

    # Get authenticated user token
    @with_auth
    def get_user(user=None):
        return user

    user = get_user()
    repo = FilesAPIRepo()
    workers = min(args.workers, len(missing_urls))

    failed = []

    def download_one(href):
        try:
            repo.stage_file_url(href, dataset_dir, user)
        except Exception as e:
            return (href, str(e))
        return None

    print(f"\nDownloading {len(missing_urls)} missing files ({workers} workers)...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, href) for href in missing_urls]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            result = future.result()
            if result is not None:
                failed.append(result)

    if failed:
        print(f"\n{len(failed)} files failed to download:")
        for href, err in failed[:10]:
            file_name = href.split("/stage/")[-1] if "/stage/" in href else href.split("//")[-1]
            print(f"  {file_name}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
        print("\nRe-run this script to retry failed downloads.")
    else:
        print(f"\nAll {len(missing_urls)} missing files downloaded successfully!")


if __name__ == "__main__":
    main()
