import logging
import tempfile
import zipfile
from pathlib import Path

import requests


# Official replacement for the retired classic bulk download:
# https://clinicaltrials.gov/data-api/about-api/api-migration
BULK_DOWNLOAD_URL = "https://clinicaltrials.gov/api/v2/studies/download?format=json.zip"


def setup_logging() -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def download_and_extract_zip(url: str, directory: Path, filename: str) -> None:
    """
    Downloads a ZIP file from the specified URL and extracts it to the given directory.

    Args:
        url (str): The URL from where to download the ZIP file.
        directory (Path): The directory to store and extract the ZIP file.
        filename (str): The filename for the downloaded ZIP file.

    Raises:
        Exception: If downloading or extracting the ZIP file fails.
    """
    directory.mkdir(parents=True, exist_ok=True)
    zip_file_path = directory / filename
    temporary_path = None

    try:
        logging.info("Downloading clinical trials from %s", url)
        with requests.get(url, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            # Retired endpoints can redirect to an HTML page with HTTP 200.
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError(f"Expected a ZIP archive, received HTML from {response.url}")
            with tempfile.NamedTemporaryFile(
                dir=directory, prefix=filename + ".", suffix=".part", delete=False
            ) as output:
                temporary_path = Path(output.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)

        # Opening the archive also rejects invalid downloads with a misleading
        # Content-Type. Preserve an existing ZIP until extraction succeeds.
        with zipfile.ZipFile(temporary_path, "r") as zip_ref:
            zip_ref.extractall(directory)
        temporary_path.replace(zip_file_path)
        logging.info("ZIP file downloaded and extracted successfully: %s", zip_file_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    """
    Main function to handle the downloading and extracting of a ZIP file.
    """
    setup_logging()
    all_cts_dir = Path('data/clinicaltrials_gov/all_cts')
    filename = 'AllAPIJSON.zip'

    try:
        download_and_extract_zip(BULK_DOWNLOAD_URL, all_cts_dir, filename)
    except Exception as e:
        logging.error("Failed to download or extract clinical trials: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
