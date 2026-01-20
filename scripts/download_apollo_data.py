"""
Download Apollo Research deception detection datasets.

Usage:
    python scripts/download_apollo_data.py --output_dir data/apollo_raw
    python scripts/download_apollo_data.py --datasets roleplaying --output_dir data/apollo_raw
"""

import argparse
import os
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APOLLO_BASE_URL = "https://raw.githubusercontent.com/ApolloResearch/deception-detection/main"

DATASET_FILES = {
    "roleplaying": f"{APOLLO_BASE_URL}/data/roleplaying/dataset.yaml",
    # Add more datasets here as needed:
    # "insider_trading": f"{APOLLO_BASE_URL}/data/insider_trading/llama-70b-3.3-generations.json",
}

def download_file(url: str, output_path: str, max_retries: int = 3) -> bool:
    """
    Download a file from URL to local path with retry logic.

    Args:
        url: Source URL
        output_path: Destination file path
        max_retries: Number of retry attempts

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Downloading {url}...")

    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            # Create parent directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save file
            with open(output_path, 'wb') as f:
                f.write(response.content)

            file_size = len(response.content) / 1024  # KB
            logger.info(f"✓ Saved to {output_path} ({file_size:.1f} KB)")
            return True

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download {url} after {max_retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {url}: {e}")
            return False

    return False

def main():
    parser = argparse.ArgumentParser(
        description="Download Apollo Research deception detection datasets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/apollo_raw",
        help="Output directory for downloaded datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["roleplaying"],
        choices=list(DATASET_FILES.keys()),
        help="Which datasets to download"
    )
    args = parser.parse_args()

    logger.info(f"{'='*70}")
    logger.info(f"Apollo Research Dataset Downloader")
    logger.info(f"{'='*70}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Datasets to download: {', '.join(args.datasets)}")
    logger.info(f"{'='*70}\n")

    downloaded = {}
    failed = []

    for dataset_name in args.datasets:
        url = DATASET_FILES[dataset_name]

        # Determine output path (preserve filename from URL)
        filename = url.split("/")[-1]
        output_path = os.path.join(args.output_dir, dataset_name, filename)

        # Download
        success = download_file(url, output_path)

        if success:
            downloaded[dataset_name] = output_path
        else:
            failed.append(dataset_name)

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"Download Summary")
    logger.info(f"{'='*70}")

    if downloaded:
        logger.info(f"\n✓ Successfully downloaded {len(downloaded)} dataset(s):")
        for name, path in downloaded.items():
            logger.info(f"  • {name}: {path}")

    if failed:
        logger.warning(f"\n✗ Failed to download {len(failed)} dataset(s):")
        for name in failed:
            logger.warning(f"  • {name}")
        return 1

    logger.info(f"\n{'='*70}")
    logger.info(f"All downloads complete!")
    logger.info(f"{'='*70}\n")

    return 0

if __name__ == "__main__":
    exit(main())
