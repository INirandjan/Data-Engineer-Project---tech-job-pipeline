"""Unit tests for the silver cleanse and gold star schema."""

from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    DateType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from src.transform.clean_jobs import (
    DELTA_CATALOG,
    DELTA_EXTENSION,
    JobTransformError,
    RawDataNotFoundError,
    UNKNOWN_COMPANY,
    UNSPECIFIED_LOCATION,
    build_star_schema,
    clean_raw_data,
    create_spark_session,
    find_latest_raw_file,
    resolve_delta_root,
    write_table,
)

BRONZE_SCHEMA = StructType(
    [
        StructField("id", LongType()),
        StructField("title", StringType()),
        StructField("company_name", StringType()),
        StructField("category", StringType()),
        StructField("publication_date", StringType()),
        StructField("candidate_required_location", StringType()),
        StructField("salary", StringType()),
        StructField("description", StringType()),
    ]
)


@pytest.fixture(scope="module")
def spark() -> SparkSession:
    """One local Spark session for every transform test in this module."""
    session = create_spark_session("tech-job-transform-tests")
    yield session
    session.stop()


def _bronze(spark: SparkSession, rows: list[tuple[object, ...]]):
    return spark.createDataFrame(rows, schema=BRONZE_SCHEMA)


def test_resolve_delta_root_local_directory() -> None:
    root = resolve_delta_root("local")
    assert root.endswith("data/processed/delta")


def test_resolve_delta_root_azure_abfss() -> None:
    root = resolve_delta_root("azure", account="jobsacct", container="lake")
    assert root == "abfss://lake@jobsacct.dfs.core.windows.net/processed/delta"


def test_resolve_delta_root_azure_requires_coordinates() -> None:
    with pytest.raises(JobTransformError, match="AZURE_STORAGE_ACCOUNT"):
        resolve_delta_root("azure", account="", container="")


def test_spark_session_enables_delta_lake(spark: SparkSession) -> None:
    assert spark.conf.get("spark.sql.extensions") == DELTA_EXTENSION
    assert spark.conf.get("spark.sql.catalog.spark_catalog") == DELTA_CATALOG


def test_find_latest_raw_file_uses_timestamp_in_the_name(tmp_path: Path) -> None:
    older = tmp_path / "jobs_raw_20260101_000000.json"
    newer = tmp_path / "jobs_raw_20260930_155148.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")

    assert find_latest_raw_file(tmp_path) == newer


def test_find_latest_raw_file_rejects_an_empty_landing_zone(tmp_path: Path) -> None:
    with pytest.raises(RawDataNotFoundError):
        find_latest_raw_file(tmp_path)


def test_clean_raw_data_strips_html_filters_titles_and_parses_dates(
    spark: SparkSession,
) -> None:
    jobs = _bronze(
        spark,
        [
            (
                1,
                "Data Engineer",
                "Acme",
                "Software Development",
                "2026-09-21T12:55:11",
                "Netherlands",
                "",
                "<p>Build&nbsp;<b>pipelines</b> &amp; models.</p>",
            ),
            (
                2,
                "Python Developer",
                "Acme",
                "Software Development",
                "2026-09-18T16:43:22",
                "USA",
                "$90k - $105k",
                "Write Python services",
            ),
            (
                3,
                "Content Reviewer",
                "TELUS Digital",
                "All others",
                "2026-09-21T12:55:11",
                "USA",
                "",
                "<div>Ignore me</div>",
            ),
        ],
    )

    silver = clean_raw_data(jobs)
    rows = {row.id: row for row in silver.collect()}

    assert set(rows) == {1, 2}
    assert silver.schema["publication_date"].dataType == TimestampType()
    assert rows[1].description == "Build pipelines & models."
    assert "<" not in rows[1].description
    assert rows[1].salary is None
    assert rows[2].salary == "$90k - $105k"
    assert rows[2].description == "Write Python services"
    formatted = {
        row.id: row.published
        for row in silver.selectExpr(
            "id",
            "date_format(publication_date, 'yyyy-MM-dd HH:mm:ss') AS published",
        ).collect()
    }
    assert formatted[1] == "2026-09-21 12:55:11"


def test_clean_raw_data_keeps_the_latest_duplicate(spark: SparkSession) -> None:
    jobs = _bronze(
        spark,
        [
            (
                9,
                "Data Engineer",
                "Acme",
                "Software Development",
                "2026-09-01T00:00:00",
                "USA",
                None,
                "Earlier posting",
            ),
            (
                9,
                "Senior Data Engineer",
                "Acme",
                "Software Development",
                "2026-09-22T08:00:00",
                "USA",
                None,
                "<p>Later posting</p>",
            ),
        ],
    )

    rows = clean_raw_data(jobs).collect()

    assert len(rows) == 1
    assert rows[0].title == "Senior Data Engineer"
    assert rows[0].description == "Later posting"


def test_build_star_schema_links_facts_to_dimensions(spark: SparkSession) -> None:
    jobs = _bronze(
        spark,
        [
            (
                1,
                "Data Engineer",
                "Beta",
                "Data",
                "2026-09-21T12:55:11",
                "Netherlands",
                "100k",
                "<p>Pipelines</p>",
            ),
            (
                2,
                "Python Developer",
                "Acme",
                "Software Development",
                "2026-09-18T16:43:22",
                "USA",
                None,
                "Services",
            ),
            (
                3,
                "Analytics Engineer",
                None,
                "Data",
                "2026-09-19T10:00:00",
                None,
                None,
                "Models",
            ),
        ],
    )
    silver = clean_raw_data(jobs)
    gold = build_star_schema(silver)
    companies = {
        row.company_name: row.company_id for row in gold["dim_company"].collect()
    }
    locations = {
        row.location_name: row.location_id for row in gold["dim_location"].collect()
    }
    facts = {row.vacancy_id: row for row in gold["fct_vacancies"].collect()}

    assert companies["Acme"] < companies["Beta"]
    assert UNKNOWN_COMPANY in companies
    assert locations["Netherlands"] < locations["USA"]
    assert UNSPECIFIED_LOCATION in locations
    assert set(facts) == {1, 2, 3}
    assert facts[2].company_id == companies["Acme"]
    assert facts[2].location_id == locations["USA"]
    assert facts[3].company_id == companies[UNKNOWN_COMPANY]
    assert facts[3].location_id == locations[UNSPECIFIED_LOCATION]
    assert facts[1].description == "Pipelines"
    assert facts[1].publication_date.isoformat() == "2026-09-21"
    assert gold["fct_vacancies"].schema["vacancy_id"].dataType == LongType()
    assert gold["fct_vacancies"].schema["company_id"].dataType == IntegerType()
    assert gold["fct_vacancies"].schema["publication_date"].dataType == DateType()
    assert (
        gold["fct_vacancies"].schema["publication_timestamp"].dataType
        == TimestampType()
    )

    joined = (
        gold["fct_vacancies"]
        .join(gold["dim_company"], "company_id", "inner")
        .join(gold["dim_location"], "location_id", "inner")
    )
    assert joined.count() == gold["fct_vacancies"].count()


def test_write_table_round_trips_delta(spark: SparkSession, tmp_path: Path) -> None:
    jobs = _bronze(
        spark,
        [
            (
                1,
                "Data Engineer",
                "Acme",
                "Software Development",
                "2026-09-21T12:55:11",
                "Netherlands",
                None,
                "<p>Pipelines</p>",
            ),
        ],
    )
    silver = clean_raw_data(jobs)
    destination = tmp_path / "silver" / "jobs"
    write_table(silver, destination)

    loaded = spark.read.format("delta").load(destination.as_posix())
    assert loaded.count() == 1
    assert loaded.collect()[0].description == "Pipelines"
