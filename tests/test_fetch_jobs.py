"""Unit tests for the Remotive raw-job extractor."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from src.extract.fetch_jobs import (
    JobApiError,
    JobStorageError,
    build_raw_output_path,
    fetch_jobs,
    main,
    save_raw_response,
)


def _response(
    status_code: int,
    body: bytes,
    url: str = "https://remotive.com/api/remote-jobs?search=Data+Engineer",
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = url
    response._content = body
    response.headers["Content-Type"] = "application/json"
    return response


def _session(response: requests.Response) -> requests.Session:
    session = requests.Session()

    def _get(*_args: object, **_kwargs: object) -> requests.Response:
        return response

    session.get = _get  # type: ignore[method-assign]
    return session


def test_fetch_jobs_returns_payload_and_sends_search_term() -> None:
    captured: dict[str, object] = {}
    body = json.dumps({"job-count": 1, "jobs": [{"id": 7}]}).encode()
    response = _response(200, body)
    session = requests.Session()

    def _get(*_args: object, **kwargs: object) -> requests.Response:
        captured.update(kwargs)
        return response

    session.get = _get  # type: ignore[method-assign]

    payload = fetch_jobs("Data Engineer", session=session)

    assert payload["job-count"] == 1
    assert captured["params"] == {"search": "Data Engineer"}
    assert captured["timeout"] == 30


def test_fetch_jobs_raises_on_http_error_status() -> None:
    session = _session(_response(503, b'{"detail":"unavailable"}'))

    with pytest.raises(JobApiError, match="HTTP 503"):
        fetch_jobs("Python", session=session)


def test_fetch_jobs_raises_on_connection_error() -> None:
    session = requests.Session()

    def _get(*_args: object, **_kwargs: object) -> requests.Response:
        raise requests.ConnectionError("name resolution failed")

    session.get = _get  # type: ignore[method-assign]

    with pytest.raises(JobApiError, match="Could not connect"):
        fetch_jobs("Python", session=session)


def test_fetch_jobs_raises_on_timeout() -> None:
    session = requests.Session()

    def _get(*_args: object, **_kwargs: object) -> requests.Response:
        raise requests.Timeout("timed out")

    session.get = _get  # type: ignore[method-assign]

    with pytest.raises(JobApiError, match="Timed out"):
        fetch_jobs("Python", session=session)


def test_fetch_jobs_raises_when_body_is_not_json() -> None:
    session = _session(_response(200, b"not-json"))

    with pytest.raises(JobApiError, match="not valid JSON"):
        fetch_jobs("Python", session=session)


def test_save_raw_response_writes_formatted_timestamped_json(tmp_path: Path) -> None:
    captured_at = datetime(2026, 9, 30, 15, 4, 5, tzinfo=timezone.utc)
    payload = {"job-count": 1, "jobs": [{"title": "Data Engineer"}]}

    destination = save_raw_response(payload, tmp_path, captured_at=captured_at)

    assert destination == tmp_path / "jobs_raw_20260930_150405.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert destination.read_text(encoding="utf-8").startswith("{\n")


def test_build_raw_output_path_converts_naive_datetime_as_utc(tmp_path: Path) -> None:
    path = build_raw_output_path(tmp_path, datetime(2026, 1, 2, 3, 4, 5))
    assert path.name == "jobs_raw_20260102_030405.json"


def test_save_raw_response_wraps_os_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_mkdir(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "mkdir", _fail_mkdir)

    with pytest.raises(JobStorageError, match="Could not write"):
        save_raw_response({"jobs": []}, tmp_path)


def test_main_returns_zero_and_writes_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"job-count": 0, "jobs": []}

    def _fake_fetch(**_kwargs: object) -> dict[str, object]:
        return body

    monkeypatch.setattr("src.extract.fetch_jobs.fetch_jobs", _fake_fetch)
    monkeypatch.setattr(
        "src.extract.fetch_jobs.load_settings",
        lambda: {
            "api_url": "https://example.test/jobs",
            "search_term": "Data Engineer",
        },
    )

    exit_code = main(["--search", "Python", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    written = list(tmp_path.glob("jobs_raw_*.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text(encoding="utf-8")) == body


def test_main_returns_error_code_when_api_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_fetch(**_kwargs: object) -> dict[str, object]:
        raise JobApiError("HTTP 500 from upstream")

    monkeypatch.setattr("src.extract.fetch_jobs.fetch_jobs", _fake_fetch)
    monkeypatch.setattr(
        "src.extract.fetch_jobs.load_settings",
        lambda: {
            "api_url": "https://example.test/jobs",
            "search_term": "Data Engineer",
        },
    )

    assert main(["--search", "Data Engineer"]) == 1


def test_main_rejects_empty_search(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.extract.fetch_jobs.load_settings",
        lambda: {
            "api_url": "https://example.test/jobs",
            "search_term": "Data Engineer",
        },
    )

    assert main(["--search", "   "]) == 2
