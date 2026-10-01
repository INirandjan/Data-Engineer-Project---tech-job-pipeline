"""Dashboard loading without PySpark or a live API call."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from src.dashboard.app import (
    SAMPLE_PATH,
    SOURCE_RAW,
    SOURCE_SAMPLE,
    UNKNOWN_COMPANY,
    UNSPECIFIED_LOCATION,
    active_parquet_files,
    resolve_vacancies,
    vacancies_from_jobs,
)


def test_dashboard_source_does_not_import_pyspark() -> None:
    """Streamlit Cloud must import the app without the repo package or a JDK."""
    source = Path("src/dashboard/app.py").read_text(encoding="utf-8")
    assert "pyspark" not in source
    assert "clean_jobs" not in source
    assert "from src" not in source
    assert "import src" not in source


def test_lightweight_transform_matches_silver_rules() -> None:
    """HTML, the title filter, duplicate ids, and blank dimensions follow silver."""
    jobs = [
        {
            "id": 7,
            "title": "Data Engineer",
            "company_name": "Acme",
            "category": "Data and Analytics",
            "publication_date": "2026-09-02T10:00:00",
            "candidate_required_location": "Europe",
            "salary": "stale",
            "description": "<p>Old</p>",
        },
        {
            "id": 7,
            "title": "Senior Data Engineer",
            "company_name": "Acme",
            "category": "Data and Analytics",
            "publication_date": "2026-09-20T10:00:00",
            "candidate_required_location": "Europe",
            "salary": "€80k",
            "description": "<p>Build <b>pipelines</b> &amp; models.</p>",
        },
        {
            "id": 8,
            "title": "Content Reviewer",
            "company_name": "Skip",
            "category": "All others",
            "publication_date": "2026-09-21T10:00:00",
            "candidate_required_location": "USA",
            "salary": "",
            "description": "Outside the keyword filter.",
        },
        {
            "id": 9,
            "title": "Python Developer",
            "company_name": " ",
            "category": "Software Development",
            "publication_date": "2026-09-11T10:00:00",
            "candidate_required_location": "",
            "salary": "",
            "description": "Blank dimensions.",
        },
    ]
    frame = vacancies_from_jobs(jobs)
    assert list(frame["vacancy_id"]) == [7, 9]
    kept = frame.loc[frame["vacancy_id"] == 7].iloc[0]
    assert kept["description"] == "Build pipelines & models."
    assert kept["salary"] == "€80k"
    blank = frame.loc[frame["vacancy_id"] == 9].iloc[0]
    assert blank["company_name"] == UNKNOWN_COMPANY
    assert blank["location_name"] == UNSPECIFIED_LOCATION


def test_delta_reader_skips_removed_parquet(tmp_path: Path) -> None:
    """An overwrite in the Delta log must not leave the old Parquet file visible."""
    table = tmp_path / "fct_vacancies"
    table.mkdir()
    connection = duckdb.connect()
    try:
        connection.execute(
            "COPY (SELECT 1::BIGINT AS vacancy_id, 'Drop' AS title) "
            f"TO '{(table / 'drop.parquet').as_posix()}' (FORMAT PARQUET)"
        )
        connection.execute(
            "COPY (SELECT 2::BIGINT AS vacancy_id, 'Keep' AS title) "
            f"TO '{(table / 'keep.parquet').as_posix()}' (FORMAT PARQUET)"
        )
    finally:
        connection.close()
    log = table / "_delta_log"
    log.mkdir()
    (log / "00000000000000000000.json").write_text(
        "\n".join(
            [
                json.dumps({"add": {"path": "drop.parquet"}}),
                json.dumps({"add": {"path": "keep.parquet"}}),
                json.dumps({"remove": {"path": "drop.parquet"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    files = active_parquet_files(table)
    assert [Path(path).name for path in files] == ["keep.parquet"]


def test_missing_lake_uses_raw_json_before_sample(tmp_path: Path) -> None:
    """Bronze JSON wins when Gold is absent, and the sample is not fetched."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    payload = {
        "jobs": [
            {
                "id": 42,
                "title": "Data Engineer",
                "company_name": "Local Bronze",
                "category": "Data and Analytics",
                "publication_date": "2026-09-19T08:00:00",
                "candidate_required_location": "Europe",
                "salary": "",
                "description": "From the raw landing zone.",
            }
        ]
    }
    (raw_dir / "jobs_raw_20260101_000000.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    frame, source = resolve_vacancies(
        delta_root=tmp_path / "no-delta",
        parquet_root=tmp_path / "no-parquet",
        raw_dir=raw_dir,
        sample_path=SAMPLE_PATH,
        allow_fetch=False,
    )
    assert source == SOURCE_RAW
    assert list(frame["vacancy_id"]) == [42]
    assert frame.iloc[0]["company_name"] == "Local Bronze"


def test_sample_loads_when_lake_and_api_are_unavailable(tmp_path: Path) -> None:
    """Streamlit Cloud has no data lake; the bundled demo must still render."""
    frame, source = resolve_vacancies(
        delta_root=tmp_path / "no-delta",
        parquet_root=tmp_path / "no-parquet",
        raw_dir=tmp_path / "no-raw",
        sample_path=SAMPLE_PATH,
        allow_fetch=False,
    )
    assert source == SOURCE_SAMPLE
    assert len(frame) == 8
    assert "Content Reviewer" not in set(frame["title"])
    assert set(frame["company_name"]) >= {UNKNOWN_COMPANY, "Northwind Analytics"}
