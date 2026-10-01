"""Land raw remote job listings from public job-board APIs.

This is step 1 of the tech-job pipeline. The default run calls Remotive,
Jobicy, and Arbeitnow for the categories dev, data, software-dev, and
backend. None of those endpoints needs an API key. The merged jobs are
written as one JSON document under ``data/raw``. Later transform jobs read
every JSON file in that directory as the bronze layer.

When these listings are shown elsewhere, link back to the source job URL
and credit the board: Remotive, Jobicy, or Arbeitnow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_API_URL = "https://remotive.com/api/remote-jobs"
JOBICY_API_URL = "https://jobicy.com/api/v2/remote-jobs"
ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"
DEFAULT_SEARCH_TERM = "Data Engineer"
DEFAULT_CATEGORIES: tuple[str, ...] = ("dev", "data", "software-dev", "backend")
DEFAULT_TIMEOUT_SECONDS = 30
JOBICY_PAGE_SIZE = 100
ARBEITNOW_MAX_PAGES = 40
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
        "jobicy_api_url": os.getenv("JOBICY_API_URL", JOBICY_API_URL),
        "arbeitnow_api_url": os.getenv("ARBEITNOW_API_URL", ARBEITNOW_API_URL),
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
        payload = _get_json(http, api_url, params, timeout_seconds)
    finally:
        if owns_session:
            http.close()

    logger.info(
        "Received %s job listing(s) for search=%r",
        _job_count(payload),
        search_term,
    )
    return payload


def _get_json(
    session: requests.Session,
    url: str,
    params: dict[str, str] | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    """GET one JSON object. Network and HTTP failures become ``JobApiError``."""
    try:
        response = session.get(
            url,
            params=params or None,
            timeout=timeout_seconds,
            headers={
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        _raise_for_status(response)
        return _parse_json_object(response)
    except requests.Timeout as exc:
        raise JobApiError(f"Timed out after {timeout_seconds}s calling {url}") from exc
    except requests.ConnectionError as exc:
        raise JobApiError(f"Could not connect to {url}") from exc
    except requests.RequestException as exc:
        raise JobApiError(f"Request to {url} failed: {exc}") from exc


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


def canonical_job_id(source: str, native_id: object) -> int:
    """Return a stable positive int that does not collide across boards."""
    digest = hashlib.sha256(f"{source}:{native_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _text(value: object) -> str:
    """Return a stripped string, or an empty string for missing values."""
    if value is None:
        return ""
    return str(value).strip()


def _publication_text(value: object) -> str:
    """Normalize a publication timestamp to an ISO-like string."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000
        return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return str(value).strip()


def _first_tag(tags: object) -> str:
    """Return the first tag when the API sent a list or a plain string."""
    if isinstance(tags, list):
        for tag in tags:
            text = _text(tag)
            if text:
                return text
        return ""
    return _text(tags)


def normalize_remotive_job(job: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Map one Remotive listing onto the shared bronze job shape."""
    if not isinstance(job, dict) or job.get("id") is None or not _text(job.get("title")):
        return None
    return {
        "id": canonical_job_id("remotive", job.get("id")),
        "source": "remotive",
        "source_job_id": _text(job.get("id")),
        "query_category": category,
        "title": _text(job.get("title")),
        "company_name": _text(job.get("company_name")),
        "category": _text(job.get("category")) or category,
        "publication_date": _publication_text(job.get("publication_date")),
        "candidate_required_location": _text(job.get("candidate_required_location")),
        "salary": _text(job.get("salary")),
        "description": _text(job.get("description")),
        "url": _text(job.get("url")),
    }


def normalize_jobicy_job(job: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Map one Jobicy listing onto the shared bronze job shape."""
    if not isinstance(job, dict) or job.get("id") is None:
        return None
    title = _text(job.get("jobTitle") or job.get("title"))
    if not title:
        return None
    return {
        "id": canonical_job_id("jobicy", job.get("id")),
        "source": "jobicy",
        "source_job_id": _text(job.get("id")),
        "query_category": category,
        "title": title,
        "company_name": _text(job.get("companyName") or job.get("company_name")),
        "category": _text(job.get("jobIndustry") or job.get("category")) or category,
        "publication_date": _publication_text(job.get("pubDate") or job.get("publication_date")),
        "candidate_required_location": _text(job.get("jobGeo") or job.get("candidate_required_location")),
        "salary": _text(job.get("salary")),
        "description": _text(job.get("jobDescription") or job.get("description")),
        "url": _text(job.get("url")),
    }


def normalize_arbeitnow_job(job: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Map one Arbeitnow listing onto the shared bronze job shape."""
    if not isinstance(job, dict):
        return None
    native_id = job.get("slug") or job.get("url") or job.get("title")
    title = _text(job.get("title"))
    if not native_id or not title:
        return None
    return {
        "id": canonical_job_id("arbeitnow", native_id),
        "source": "arbeitnow",
        "source_job_id": _text(native_id),
        "query_category": category,
        "title": title,
        "company_name": _text(job.get("company_name")),
        "category": _first_tag(job.get("tags")) or category,
        "publication_date": _publication_text(job.get("created_at") or job.get("publication_date")),
        "candidate_required_location": _text(job.get("location")),
        "salary": _text(job.get("salary")),
        "description": _text(job.get("description")),
        "url": _text(job.get("url")),
    }


def _remember_job(
    jobs: list[dict[str, Any]],
    seen_ids: set[int],
    seen_pairs: set[tuple[str, str]],
    job: dict[str, Any] | None,
) -> None:
    """Append ``job`` unless the same id or title/company pair is already kept."""
    if job is None:
        return
    pair = (job["title"].casefold(), job["company_name"].casefold())
    if job["id"] in seen_ids or pair in seen_pairs:
        return
    seen_ids.add(job["id"])
    seen_pairs.add(pair)
    jobs.append(job)


def jobicy_queries(category: str) -> list[dict[str, str]]:
    """Return Jobicy query strings for one requested category."""
    count = str(JOBICY_PAGE_SIZE)
    if category == "dev":
        return [{"count": count, "industry": "dev"}]
    if category == "data":
        return [
            {"count": count, "tag": "data"},
            {"count": count, "industry": "data-science"},
        ]
    if category == "software-dev":
        return [
            {"count": count, "tag": "software-dev"},
            {"count": count, "industry": "dev"},
        ]
    if category == "backend":
        return [{"count": count, "tag": "backend"}]
    return [{"count": count, "tag": category}]


def _extend_source(
    jobs: list[dict[str, Any]],
    seen_ids: set[int],
    seen_pairs: set[tuple[str, str]],
    batch: list[dict[str, Any] | None],
) -> int:
    """Remember a batch and return how many new rows were kept."""
    before = len(jobs)
    for job in batch:
        _remember_job(jobs, seen_ids, seen_pairs, job)
    return len(jobs) - before


def collect_catalog(
    categories: tuple[str, ...] | list[str] = DEFAULT_CATEGORIES,
    remotive_api_url: str = DEFAULT_API_URL,
    jobicy_api_url: str = JOBICY_API_URL,
    arbeitnow_api_url: str = ARBEITNOW_API_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    session: requests.Session | None = None,
    pause_seconds: float = 0.25,
) -> dict[str, Any]:
    """Fetch every configured board and category into one bronze document.

    A failed call is recorded and skipped. The run fails only when every
    call failed or none of them returned a job.
    """
    http = session or requests.Session()
    owns_session = session is None
    jobs: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    seen_pairs: set[tuple[str, str]] = set()
    attempts: list[dict[str, Any]] = []
    failures = 0
    calls = 0

    def _pause() -> None:
        if pause_seconds > 0 and calls:
            time.sleep(pause_seconds)

    try:
        for category in categories:
            _pause()
            calls += 1
            try:
                payload = _get_json(
                    http,
                    remotive_api_url,
                    {"category": category, "search": category},
                    timeout_seconds,
                )
                added = _extend_source(
                    jobs,
                    seen_ids,
                    seen_pairs,
                    [
                        normalize_remotive_job(job, category)
                        for job in payload.get("jobs") or []
                        if isinstance(job, dict)
                    ],
                )
                attempts.append({"source": "remotive", "category": category, "added": added})
                logger.info("Remotive category %s added %s new job(s)", category, added)
            except JobApiError as exc:
                failures += 1
                attempts.append(
                    {"source": "remotive", "category": category, "error": str(exc)}
                )
                logger.warning("Remotive category %s failed: %s", category, exc)

            for params in jobicy_queries(category):
                _pause()
                calls += 1
                try:
                    payload = _get_json(http, jobicy_api_url, params, timeout_seconds)
                    added = _extend_source(
                        jobs,
                        seen_ids,
                        seen_pairs,
                        [
                            normalize_jobicy_job(job, category)
                            for job in payload.get("jobs") or []
                            if isinstance(job, dict)
                        ],
                    )
                    attempts.append(
                        {"source": "jobicy", "category": category, "params": params, "added": added}
                    )
                    logger.info("Jobicy %s added %s new job(s)", params, added)
                except JobApiError as exc:
                    failures += 1
                    attempts.append(
                        {
                            "source": "jobicy",
                            "category": category,
                            "params": params,
                            "error": str(exc),
                        }
                    )
                    logger.warning("Jobicy %s failed: %s", params, exc)

        page = 1
        while page <= ARBEITNOW_MAX_PAGES:
            _pause()
            calls += 1
            payload = None
            for attempt in range(2):
                try:
                    payload = _get_json(
                        http,
                        arbeitnow_api_url,
                        {"page": str(page)},
                        timeout_seconds,
                    )
                    break
                except JobApiError as exc:
                    rate_limited = "429" in str(exc)
                    if rate_limited and attempt == 0:
                        logger.warning("Arbeitnow page %s was rate limited; retrying", page)
                        time.sleep(3)
                        continue
                    failures += 1
                    attempts.append({"source": "arbeitnow", "page": page, "error": str(exc)})
                    logger.warning("Arbeitnow page %s failed: %s", page, exc)
                    payload = None
                    break
            if payload is None:
                break
            batch = payload.get("data") if isinstance(payload.get("data"), list) else []
            added = _extend_source(
                jobs,
                seen_ids,
                seen_pairs,
                [normalize_arbeitnow_job(job, "dev") for job in batch if isinstance(job, dict)],
            )
            attempts.append({"source": "arbeitnow", "page": page, "added": added})
            logger.info("Arbeitnow page %s added %s new job(s)", page, added)
            next_url = (payload.get("links") or {}).get("next") if isinstance(payload.get("links"), dict) else None
            if not batch or not next_url:
                break
            page += 1
    finally:
        if owns_session:
            http.close()

    if not jobs:
        raise JobApiError(
            f"No vacancies collected from Remotive, Jobicy, or Arbeitnow ({failures} failed call(s))"
        )
    logger.info("Collected %s unique vacancy listing(s)", len(jobs))
    return {
        "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "categories": list(categories),
        "sources": ["remotive", "jobicy", "arbeitnow"],
        "attempts": attempts,
        "job-count": len(jobs),
        "jobs": jobs,
    }


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
            "Download raw remote job listings from Remotive, Jobicy, and Arbeitnow."
        )
    )
    parser.add_argument(
        "--search",
        default=None,
        help=(
            "Optional Remotive-only keyword. Omit this flag to pull "
            "dev, data, software-dev, and backend from all three boards."
        ),
    )
    parser.add_argument(
        "--api-url",
        default=settings["api_url"],
        help="Remotive endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--jobicy-api-url",
        default=settings.get("jobicy_api_url") or JOBICY_API_URL,
        help="Jobicy endpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--arbeitnow-api-url",
        default=settings.get("arbeitnow_api_url") or ARBEITNOW_API_URL,
        help="Arbeitnow endpoint (default: %(default)s).",
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
    try:
        if args.search is not None:
            search_term = str(args.search).strip()
            if not search_term:
                logger.error("Search term is empty")
                return 2
            run_extraction(
                search_term=search_term,
                api_url=args.api_url,
                output_dir=args.output_dir,
            )
        else:
            payload = collect_catalog(
                remotive_api_url=args.api_url,
                jobicy_api_url=args.jobicy_api_url,
                arbeitnow_api_url=args.arbeitnow_api_url,
            )
            save_raw_response(payload, args.output_dir)
    except JobExtractionError:
        logger.exception("Job extraction failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
