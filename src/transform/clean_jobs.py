"""Build the silver and gold layers for remote IT vacancies.

Bronze JSON from ``data/raw`` is cleaned into a silver vacancy table, then
modeled as a star schema:

* ``dim_company`` and ``dim_location`` hold generated surrogate keys
* ``fct_vacancies`` stores one row per vacancy with foreign keys

Tables are written under ``data/processed``. Delta Lake is used when the
``delta`` package is installed; otherwise the writer falls back to Parquet.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, TimestampNTZType, TimestampType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_FILE_PATTERN = "jobs_raw_*.json"
TITLE_KEYWORDS: tuple[str, ...] = ("data", "engineer", "developer", "python")
UNKNOWN_COMPANY = "Unknown"
UNSPECIFIED_LOCATION = "Unspecified"
HTML_TAG_PATTERN = r"<[^>]+>"
HTML_ENTITIES: tuple[tuple[str, str], ...] = (
    ("&nbsp;", " "),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&amp;", "&"),
)


class JobTransformError(Exception):
    """Base error for a failed silver/gold build."""


class RawDataNotFoundError(JobTransformError):
    """No bronze JSON file was found in the raw landing zone."""


_WINUTILS_STUB_SOURCE = """
using System;

internal static class Program
{
    private static int Main(string[] args)
    {
        if (args.Length > 0 &&
            string.Equals(args[0], "ls", StringComparison.OrdinalIgnoreCase))
        {
            Console.WriteLine("-rwxrwxrwx 1 spark spark 0 file");
        }
        return 0;
    }
}
"""


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a stream handler when the process has not configured logging yet."""
    logging.getLogger("py4j").setLevel(logging.ERROR)
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def delta_is_available() -> bool:
    """Return whether the Delta Lake Python package can be imported."""
    return importlib.util.find_spec("delta") is not None


def assert_java_available() -> None:
    """Fail fast when the JVM Spark needs is not installed."""
    if os.environ.get("JAVA_HOME") or shutil.which("java"):
        return
    local_jdks = Path(os.environ.get("LOCALAPPDATA", "")) / "jdks"
    discovered = (
        sorted(local_jdks.glob("jdk-*/bin/java.exe")) if local_jdks.exists() else []
    )
    if discovered:
        os.environ["JAVA_HOME"] = str(discovered[-1].parent.parent)
        logger.info("Using JAVA_HOME=%s", os.environ["JAVA_HOME"])
        return
    raise JobTransformError(
        "Java 17 or newer is required to run PySpark. "
        "Install a JDK and set JAVA_HOME."
    )


def _spark_python_executable() -> str:
    """Return a Python path Spark's Windows cmd scripts can launch.

    Those scripts split on spaces, and this repository path contains them.
    The 8.3 short path points at the same interpreter without spaces.
    """
    executable = str(Path(sys.executable))
    if os.name != "nt" or " " not in executable:
        return executable
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    length = ctypes.windll.kernel32.GetShortPathNameW(
        executable, buffer, len(buffer)
    )
    short_path = buffer.value if length else ""
    if short_path and " " not in short_path:
        return short_path
    return executable


def _local_tool_dir() -> Path:
    """Return a user-local directory whose path contains no spaces."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("TEMP") or str(Path.home())
    return Path(base) / "tech-job-pipeline"


def ensure_windows_hadoop_home() -> None:
    """Point Hadoop at a local ``winutils.exe`` so Parquet writes work on Windows.

    Spark's Hadoop client refuses to create output directories unless
    ``HADOOP_HOME/bin/winutils.exe`` exists. A full Hadoop distribution is
    unnecessary for this local pipeline: ``chmod`` and ``ls`` only need to
    succeed. An existing ``HADOOP_HOME`` is left untouched.
    """
    if os.name != "nt" or os.environ.get("HADOOP_HOME"):
        return

    home = _local_tool_dir() / "hadoop"
    executable = home / "bin" / "winutils.exe"
    if not executable.exists():
        executable.parent.mkdir(parents=True, exist_ok=True)
        source = home / "winutils_stub.cs"
        source.write_text(_WINUTILS_STUB_SOURCE, encoding="utf-8")
        compiler = Path(r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")
        if not compiler.exists():
            compiler = Path(r"C:\Windows\Microsoft.NET\Framework\v4.0.30319\csc.exe")
        if not compiler.exists():
            raise JobTransformError(
                "PySpark on Windows needs winutils.exe. Install the .NET "
                "Framework compiler, or set HADOOP_HOME to a directory that "
                "contains bin/winutils.exe."
            )
        completed = subprocess.run(
            [
                str(compiler),
                "/nologo",
                "/optimize+",
                f"/out:{executable}",
                str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not executable.exists():
            detail = (completed.stdout + completed.stderr).strip()
            raise JobTransformError(
                f"Could not create a local winutils stub: {detail}"
            )
        logger.info("Created a local winutils stub at %s", executable)

    os.environ["HADOOP_HOME"] = str(home)
    logger.info("Using HADOOP_HOME=%s", home)


_LOCAL_FILE_SYSTEM_SOURCE = """
package org.techjob;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.RawLocalFileSystem;
import org.apache.hadoop.fs.permission.FsPermission;

/**
 * Local filesystem that lists files with java.io.File.
 * Hadoop 3.5 on Windows calls a native access() during listStatus, which
 * fails unless hadoop.dll is installed. This pipeline only writes a local lake.
 */
public class LocalFileSystem extends RawLocalFileSystem {
    @Override
    public FileStatus[] listStatus(Path path) throws IOException {
        File local = pathToFile(path);
        if (!local.exists()) {
            throw new FileNotFoundException(path.toString());
        }
        if (local.isFile()) {
            return new FileStatus[] { statusOf(local) };
        }
        File[] children = local.listFiles();
        if (children == null) {
            return new FileStatus[0];
        }
        FileStatus[] result = new FileStatus[children.length];
        for (int index = 0; index < children.length; index++) {
            result[index] = statusOf(children[index]);
        }
        return result;
    }

    @Override
    public FileStatus getFileStatus(Path path) throws IOException {
        File local = pathToFile(path);
        if (!local.exists()) {
            throw new FileNotFoundException(path.toString());
        }
        return statusOf(local);
    }

    @Override
    public void setPermission(Path path, FsPermission permission) {
        // NTFS permissions are outside this local batch.
    }

    private FileStatus statusOf(File file) {
        long length = file.isDirectory() ? 0L : file.length();
        return new FileStatus(
            length,
            file.isDirectory(),
            1,
            128L * 1024L * 1024L,
            file.lastModified(),
            new Path(file.toURI())
        );
    }
}
"""


def _hadoop_client_jar() -> Path:
    """Return the Hadoop client API jar shipped with PySpark."""
    import pyspark

    jars = sorted((Path(pyspark.__file__).parent / "jars").glob("hadoop-client-api-*.jar"))
    if not jars:
        raise JobTransformError("hadoop-client-api jar was not found in the PySpark install")
    return jars[-1]


def _javac_executable() -> Path:
    """Return javac from JAVA_HOME or PATH."""
    java_home = os.environ.get("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / "javac.exe"
        if candidate.exists():
            return candidate
    found = shutil.which("javac")
    if found:
        return Path(found)
    raise JobTransformError(
        "javac was not found. Set JAVA_HOME to a JDK 17 or newer install."
    )


def ensure_windows_local_filesystem() -> str | None:
    """Compile a pure-Java local filesystem and return its classpath.

    Returns None on non-Windows platforms, where Hadoop's built-in local
    filesystem can list directories without a native library.
    """
    if os.name != "nt":
        return None

    classes = _local_tool_dir() / "fs-classes"
    marker = classes / "org" / "techjob" / "LocalFileSystem.class"
    if not marker.exists():
        source_dir = _local_tool_dir() / "fs-src" / "org" / "techjob"
        source_dir.mkdir(parents=True, exist_ok=True)
        source = source_dir / "LocalFileSystem.java"
        source.write_text(_LOCAL_FILE_SYSTEM_SOURCE, encoding="utf-8")
        classes.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            [
                str(_javac_executable()),
                "--release",
                "17",
                "-encoding",
                "UTF-8",
                "-cp",
                str(_hadoop_client_jar()),
                "-d",
                str(classes),
                str(source),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not marker.exists():
            detail = (completed.stdout + completed.stderr).strip()
            raise JobTransformError(
                f"Could not compile the local filesystem helper: {detail}"
            )
        logger.info("Compiled local filesystem helper at %s", marker)
    return classes.as_posix()


def create_spark_session(
    app_name: str = "tech-job-transform",
    *,
    enable_delta: bool = False,
) -> SparkSession:
    """Start a local Spark session for the batch transform."""
    assert_java_available()
    ensure_windows_hadoop_home()
    local_filesystem = ensure_windows_local_filesystem()
    python_executable = _spark_python_executable()
    for variable in ("PYSPARK_PYTHON", "PYSPARK_DRIVER_PYTHON"):
        current = os.environ.get(variable)
        if not current or " " in current:
            os.environ[variable] = python_executable

    warehouse = _local_tool_dir() / "spark-warehouse"
    builder = (
        SparkSession.builder.master("local[1]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.warehouse.dir", warehouse.as_posix())
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
    )
    if local_filesystem:
        builder = (
            builder.config("spark.driver.extraClassPath", local_filesystem)
            .config("spark.executor.extraClassPath", local_filesystem)
            .config("spark.hadoop.fs.file.impl", "org.techjob.LocalFileSystem")
        )
    if enable_delta:
        builder = builder.config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        ).config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def find_latest_raw_file(raw_dir: Path) -> Path:
    """Return the newest ``jobs_raw_<timestamp>.json`` file in ``raw_dir``.

    The extractor encodes UTC time in the file name, so lexicographic order
    matches chronological order.
    """
    candidates = sorted(
        path
        for path in raw_dir.glob(RAW_FILE_PATTERN)
        if path.is_file()
    )
    if not candidates:
        raise RawDataNotFoundError(
            f"No files matching {RAW_FILE_PATTERN} in {raw_dir}"
        )
    return candidates[-1]


def read_raw_jobs(spark: SparkSession, path: Path) -> DataFrame:
    """Read the Remotive ``jobs`` array from one bronze JSON document."""
    logger.info("Reading bronze file %s", path)
    raw = spark.read.option("multiLine", "true").json(path.as_posix())
    if "jobs" not in raw.columns:
        raise JobTransformError(f"{path.name} does not contain a jobs array")
    return raw.select(F.explode("jobs").alias("job")).select("job.*")


def strip_html(column: Column) -> Column:
    """Remove HTML tags and common entities, then collapse whitespace."""
    text = F.regexp_replace(column, HTML_TAG_PATTERN, " ")
    for entity, replacement in HTML_ENTITIES:
        text = F.regexp_replace(text, entity, replacement)
    collapsed = F.trim(F.regexp_replace(text, r"\s+", " "))
    return F.when(collapsed == "", None).otherwise(collapsed)


def _clean_text(column: Column) -> Column:
    """Trim a text column and turn blanks into null."""
    collapsed = F.trim(F.regexp_replace(column, r"\s+", " "))
    return F.when(collapsed == "", None).otherwise(collapsed)


def _optional_column(df: DataFrame, name: str) -> Column:
    """Return ``name`` when present, otherwise a null string column."""
    if name in df.columns:
        return F.col(name)
    return F.lit(None).cast("string")


def _publication_timestamp(jobs: DataFrame) -> Column:
    """Parse ``publication_date`` into a timestamp, whatever type Spark inferred."""
    if "publication_date" not in jobs.columns:
        return F.lit(None).cast("timestamp")

    column = F.col("publication_date")
    data_type = jobs.schema["publication_date"].dataType
    if isinstance(data_type, (TimestampType, TimestampNTZType, DateType)):
        return column.cast("timestamp")

    text = column.cast("string")
    return F.coalesce(
        F.to_timestamp(text, "yyyy-MM-dd'T'HH:mm:ss"),
        F.to_timestamp(text, "yyyy-MM-dd'T'HH:mm:ss.SSSSSS"),
        F.to_timestamp(text, "yyyy-MM-dd HH:mm:ss"),
        F.to_timestamp(text),
    )


def _title_is_relevant(title: Column) -> Column:
    """True when the title contains one of the IT role keywords."""
    lowered = F.lower(title)
    match = F.lit(False)
    for keyword in TITLE_KEYWORDS:
        match = match | lowered.contains(keyword)
    return match


def clean_raw_data(jobs: DataFrame) -> DataFrame:
    """Project, clean, and filter bronze jobs into the silver vacancy table.

    The result keeps source identifiers and text. Generated dimension keys
    are added later in :func:`build_star_schema`.
    """
    logger.info(
        "Cleaning bronze jobs; title must contain one of: %s",
        ", ".join(TITLE_KEYWORDS),
    )
    cleaned = jobs.select(
        _optional_column(jobs, "id").cast("long").alias("id"),
        _clean_text(_optional_column(jobs, "title")).alias("title"),
        _clean_text(_optional_column(jobs, "company_name")).alias("company_name"),
        _clean_text(_optional_column(jobs, "category")).alias("category"),
        _publication_timestamp(jobs).alias("publication_date"),
        _clean_text(_optional_column(jobs, "candidate_required_location")).alias(
            "candidate_required_location"
        ),
        _clean_text(_optional_column(jobs, "salary")).alias("salary"),
        strip_html(_optional_column(jobs, "description")).alias("description"),
    )
    relevant = cleaned.where(_title_is_relevant(F.col("title"))).where(
        F.col("id").isNotNull()
    )
    latest = Window.partitionBy("id").orderBy(
        F.col("publication_date").desc_nulls_last()
    )
    return (
        relevant.withColumn("_rank", F.row_number().over(latest))
        .where(F.col("_rank") == 1)
        .drop("_rank")
        .select(
            "id",
            "title",
            "company_name",
            "category",
            "publication_date",
            "candidate_required_location",
            "salary",
            "description",
        )
    )


def build_star_schema(silver: DataFrame) -> dict[str, DataFrame]:
    """Split silver vacancies into company, location, and fact tables.

    Surrogate keys are ``row_number`` values ordered by the business name.
    They are deterministic for a given input and are rebuilt on every full
    refresh. Missing company or location values become ``Unknown`` and
    ``Unspecified`` so every fact row has both foreign keys.

    A location value is the published requirement string, including a
    comma-separated region list. Splitting that list into a bridge table
    is left for a later model.
    """
    logger.info("Building gold star schema from silver vacancies")
    prepared = silver.select(
        F.col("id").alias("vacancy_id"),
        F.coalesce(F.col("company_name"), F.lit(UNKNOWN_COMPANY)).alias(
            "company_name"
        ),
        F.coalesce(
            F.col("candidate_required_location"),
            F.lit(UNSPECIFIED_LOCATION),
        ).alias("location_name"),
        F.to_date("publication_date").alias("publication_date"),
        F.col("publication_date").alias("publication_timestamp"),
        "title",
        "description",
        "category",
        "salary",
    )

    dim_company = (
        prepared.select("company_name")
        .distinct()
        .withColumn(
            "company_id",
            F.row_number().over(Window.orderBy("company_name")).cast("int"),
        )
        .select("company_id", "company_name")
    )
    dim_location = (
        prepared.select("location_name")
        .distinct()
        .withColumn(
            "location_id",
            F.row_number().over(Window.orderBy("location_name")).cast("int"),
        )
        .select("location_id", "location_name")
    )
    fct_vacancies = (
        prepared.join(dim_company, "company_name", "inner")
        .join(dim_location, "location_name", "inner")
        .select(
            "vacancy_id",
            "company_id",
            "location_id",
            "publication_date",
            "publication_timestamp",
            "title",
            "description",
            "category",
            "salary",
        )
    )
    return {
        "dim_company": dim_company,
        "dim_location": dim_location,
        "fct_vacancies": fct_vacancies,
    }


def choose_format(requested: str | None) -> str:
    """Pick Delta when requested or installed, otherwise Parquet."""
    if requested == "parquet":
        return "parquet"
    if requested == "delta" or (requested is None and delta_is_available()):
        if not delta_is_available():
            raise JobTransformError(
                "delta-spark is not installed. Re-run with --format parquet."
            )
        return "delta"
    logger.info("delta-spark is not installed; writing Parquet")
    return "parquet"


def write_table(frame: DataFrame, destination: Path, file_format: str) -> None:
    """Overwrite one table directory in ``file_format``."""
    writer = frame.write.mode("overwrite")
    target = destination.as_posix()
    if file_format == "delta":
        writer.format("delta").save(target)
        return
    if file_format != "parquet":
        raise JobTransformError(f"Unsupported table format: {file_format}")
    writer.parquet(target)


def run_transformation(
    spark: SparkSession,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    file_format: str = "parquet",
) -> dict[str, int]:
    """Read the latest bronze file and write silver plus gold tables.

    Returns a mapping of output-relative path to row count.
    """
    latest = find_latest_raw_file(raw_dir)
    raw_jobs = read_raw_jobs(spark, latest)
    silver = clean_raw_data(raw_jobs).cache()
    gold = build_star_schema(silver)
    tables: dict[str, DataFrame] = {
        "silver/jobs": silver,
        "gold/dim_company": gold["dim_company"],
        "gold/dim_location": gold["dim_location"],
        "gold/fct_vacancies": gold["fct_vacancies"],
    }

    counts: dict[str, int] = {}
    try:
        for relative_path, frame in tables.items():
            materialized = frame.cache()
            row_count = materialized.count()
            destination = output_dir / relative_path
            logger.info(
                "Writing %s row(s) as %s to %s",
                row_count,
                file_format,
                destination,
            )
            write_table(materialized, destination, file_format)
            counts[relative_path] = row_count
            if materialized is not silver:
                materialized.unpersist()
    finally:
        silver.unpersist()

    if counts.get("silver/jobs", 0) == 0:
        logger.warning("Silver table is empty after the title filter")
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI for one silver/gold batch."""
    parser = argparse.ArgumentParser(
        description="Clean raw job JSON into silver and gold Parquet tables."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Bronze landing directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed lake directory (default: %(default)s).",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "delta"),
        default=None,
        help="Output format. Defaults to delta when installed, else parquet.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one transformation. Returns a process exit code."""
    configure_logging()
    args = parse_args(argv)
    try:
        file_format = choose_format(args.format)
        spark = create_spark_session(enable_delta=file_format == "delta")
        try:
            run_transformation(
                spark,
                raw_dir=args.raw_dir,
                output_dir=args.output_dir,
                file_format=file_format,
            )
        finally:
            spark.stop()
    except JobTransformError:
        logger.exception("Job transformation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
