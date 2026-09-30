"""Land raw remote job listings from the Remotive public API.

This is step 1 of the tech-job pipeline. The script requests listings for a
keyword, then writes the untouched JSON body to ``data/raw``. Later transform
jobs should read those files as the bronze layer.

Remotive does not require an API key. When these listings are shown elsewhere,
link back to the Remotive job URL and credit Remotive as the source:
https://remotive.com/remote-jobs/api
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_API_URL = "https://remotive.com/api/remote-jobs"
DEFAULT_SEARCH_TERM = "Data Engineer"
DEFAULT_TIMEOUT_SECONDS = 30
USER_AGENT = "tech-job-pipeline/0.1 (educational data-engineering project)"


class JobExtractionError(Exception):
    """Base error for a failed extraction run."""


class JobApiError(JobExtractionError):
    """The jobs API could not be reached or returned an unusable payload."""


class JobStorageError(JobExtractionError):
    """The raw response could not be written to disk."""


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a stream handler when the process has not configured logging yet."""
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def load_settings() -> dict[str, str | None]:
    """Read optional overrides from the project ``.env`` file."""
    load_dotenv(PROJECT_ROOT / ".env")
    return {
        "api_url": os.getenv("REMOTIVE_API_URL", DEFAULT_API_URL),
        "search_term": os.getenv("JOB_SEARCH_TERM", DEFAULT_SEARCH_TERM),
    }


def fetch_jobs(
    search_term: str,
    api_url: str = DEFAULT_API_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Return the parsed JSON body for ``search_term``.

    The public endpoint matches the keyword as a case-insensitive substring
    of the title and description, and currently returns the full match set.
    This function does not trim that payload: the caller stores it raw.

    Args:
        search_term: Keyword matched against title and description.
        api_url: Remotive jobs endpoint.
        timeout_seconds: Socket timeout for the HTTP call.
        session: Optional session, useful for tests and connection reuse.

    Raises:
        JobApiError: On timeouts, connection failures, HTTP error statuses,
            or a body that is not a JSON object.
    """
    params: dict[str, str] = {"search": search_term}

    http = session or requests.Session()
    owns_session = session is None
    try:
        logger.info("Requesting jobs from %s (search=%r)", api_url, search_term)
        response = http.get(
            api_url,
            params=params,
            timeout=timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        _raise_for_status(response)
        payload = _parse_json_object(response)
    except requests.Timeout as exc:
        raise JobApiError(
            f"Timed out after {timeout_seconds}s calling {api_url}"
        ) from exc
    except requests.ConnectionError as exc:
        raise JobApiError(f"Could not connect to {api_url}") from exc
    except requests.RequestException as exc:
        raise JobApiError(f"Request to {api_url} failed: {exc}") from exc
    finally:
        if owns_session:
            http.close()

    logger.info(
        "Received %s job listing(s) for search=%r",
        _job_count(payload),
        search_term,
    )
    return payload


def _raise_for_status(response: requests.Response) -> None:
    """Translate an HTTP error status into ``JobApiError``."""
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        preview = response.text[:300].replace("\n", " ")
        raise JobApiError(
            f"HTTP {response.status_code} from {response.url}: {preview}"
        ) from exc


def _parse_json_object(response: requests.Response) -> dict[str, Any]:
    """Decode a JSON object from ``response``."""
    try:
        payload = response.json()
    except ValueError as exc:
        raise JobApiError("API response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise JobApiError(
            f"Expected a JSON object, received {type(payload).__name__}"
        )
    return payload


def _job_count(payload: dict[str, Any]) -> Any:
    """Prefer the API count, then fall back to the jobs array length."""
    jobs = payload.get("jobs")
    if "job-count" in payload:
        return payload["job-count"]
    if isinstance(jobs, list):
        return len(jobs)
    return "unknown"


def build_raw_output_path(
    output_dir: Path,
    captured_at: datetime | None = None,
) -> Path:
    """Return ``jobs_raw_<YYYYMMDD_HHMMSS>.json`` in UTC."""
    moment = captured_at or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    stamp = moment.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return output_dir / f"jobs_raw_{stamp}.json"


def save_raw_response(
    payload: dict[str, Any],
    output_dir: Path,
    captured_at: datetime | None = None,
) -> Path:
    """Write ``payload`` as indented UTF-8 JSON and return the file path.

    Raises:
        JobStorageError: If the directory or file cannot be created.
    """
    destination = build_raw_output_path(output_dir, captured_at)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise JobStorageError(
            f"Could not write raw response to {destination}"
        ) from exc

    logger.info("Saved raw response to %s", destination)
    return destination


def run_extraction(
    search_term: str,
    api_url: str = DEFAULT_API_URL,
    output_dir: Path = DEFAULT_RAW_DIR,
    session: requests.Session | None = None,
    captured_at: datetime | None = None,
) -> Path:
    """Fetch listings and persist the untouched API payload."""
    payload = fetch_jobs(
        search_term=search_term,
        api_url=api_url,
        session=session,
    )
    return save_raw_response(payload, output_dir, captured_at=captured_at)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI, with environment values as defaults."""
    settings = load_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Download raw remote job listings from the Remotive public API."
        )
    )
    parser.add_argument(
        "--search",
        default=settings["search_term"],
        help=(
            "Keyword matched against job title and description "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--api-url",
        default=settings["api_url"],
        help="Jobs endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Directory for the raw JSON landing file.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one extraction. Returns a process exit code."""
    configure_logging()
    args = parse_args(argv)
    search_term = str(args.search).strip()
    if not search_term:
        logger.error("Search term is empty")
        return 2

    try:
        run_extraction(
            search_term=search_term,
            api_url=args.api_url,
            output_dir=args.output_dir,
        )
    except JobExtractionError:
        logger.exception("Job extraction failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
