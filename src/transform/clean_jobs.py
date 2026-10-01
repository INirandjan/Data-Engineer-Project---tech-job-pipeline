"""Build the silver and gold layers for remote IT vacancies.

Bronze JSON from ``data/raw`` is cleaned into a silver vacancy table, then
modeled as a star schema:

* ``dim_company`` and ``dim_location`` hold generated surrogate keys
* ``fct_vacancies`` stores one row per vacancy with foreign keys

Tables are Delta Lake tables. ``STORAGE_TYPE=local`` writes them under
``data/processed/delta``. ``STORAGE_TYPE=azure`` writes the same layout to
``abfss://<container>@<account>.dfs.core.windows.net/processed/delta``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from delta.pip_utils import configure_spark_with_delta_pip
from dotenv import load_dotenv
from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, TimestampNTZType, TimestampType

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_DELTA_ROOT = DEFAULT_PROCESSED_DIR / "delta"
GOLD_SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "gold_jobs_sample.parquet"
DELTA_EXTENSION = "io.delta.sql.DeltaSparkSessionExtension"
DELTA_CATALOG = "org.apache.spark.sql.delta.catalog.DeltaCatalog"
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


def _env_value(explicit: str | None, name: str) -> str:
    """Prefer an explicit argument, otherwise read ``name`` from the environment."""
    if explicit is not None:
        return explicit.strip()
    return (os.getenv(name) or "").strip()


def load_storage_settings() -> None:
    """Load ``.env`` without overriding variables already set in the process."""
    load_dotenv(PROJECT_ROOT / ".env")


def resolve_delta_root(
    storage_type: str | None = None,
    *,
    account: str | None = None,
    container: str | None = None,
) -> str:
    """Return the Delta root for local disk or Azure Data Lake Storage.

    ``STORAGE_TYPE=local`` uses ``data/processed/delta``.
    ``STORAGE_TYPE=azure`` uses
    ``abfss://<container>@<account>.dfs.core.windows.net/processed/delta``.
    """
    load_storage_settings()
    selected = _env_value(storage_type, "STORAGE_TYPE").lower() or "local"
    if selected == "local":
        return DEFAULT_DELTA_ROOT.as_posix()
    if selected != "azure":
        raise JobTransformError(
            f"Unsupported STORAGE_TYPE {selected!r}. Use 'local' or 'azure'."
        )

    account_name = _env_value(account, "AZURE_STORAGE_ACCOUNT")
    container_name = _env_value(container, "AZURE_STORAGE_CONTAINER")
    if not account_name or not container_name:
        raise JobTransformError(
            "STORAGE_TYPE=azure requires AZURE_STORAGE_ACCOUNT and "
            "AZURE_STORAGE_CONTAINER."
        )
    return (
        f"abfss://{container_name}@{account_name}.dfs.core.windows.net/"
        "processed/delta"
    )


def _configure_azure_storage(builder: SparkSession.Builder) -> tuple[SparkSession.Builder, list[str]]:
    """Add ABFS credentials when the run targets Azure storage.

    Returns the builder and any extra Maven packages the session needs.
    A shared key is optional: Azure Databricks can supply a managed identity
    instead.
    """
    load_storage_settings()
    if os.getenv("STORAGE_TYPE", "local").strip().lower() != "azure":
        return builder, []

    account = os.getenv("AZURE_STORAGE_ACCOUNT", "").strip()
    key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "").strip()
    packages = [f"org.apache.hadoop:hadoop-azure:{_hadoop_version()}"]
    if account and key:
        host = f"{account}.dfs.core.windows.net"
        builder = builder.config(
            f"spark.hadoop.fs.azure.account.auth.type.{host}",
            "SharedKey",
        ).config(
            f"spark.hadoop.fs.azure.account.key.{host}",
            key,
        )
        logger.info("Configured shared-key auth for abfss on %s", host)
    else:
        logger.info(
            "STORAGE_TYPE=azure without a shared key; "
            "the session expects managed identity or workspace credentials."
        )
    return builder, packages


def _hadoop_version() -> str:
    """Return the Hadoop version bundled with this PySpark install."""
    stem = _hadoop_client_jar().name.removesuffix(".jar")
    return stem.rsplit("-", 1)[-1]


def create_spark_session(app_name: str = "tech-job-transform") -> SparkSession:
    """Start a local Spark session with the Delta Lake extensions enabled."""
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
        .config("spark.sql.extensions", DELTA_EXTENSION)
        .config("spark.sql.catalog.spark_catalog", DELTA_CATALOG)
    )
    if local_filesystem:
        builder = (
            builder.config("spark.driver.extraClassPath", local_filesystem)
            .config("spark.executor.extraClassPath", local_filesystem)
            .config("spark.hadoop.fs.file.impl", "org.techjob.LocalFileSystem")
        )
    builder, extra_packages = _configure_azure_storage(builder)
    builder = configure_spark_with_delta_pip(
        builder,
        extra_packages=extra_packages or None,
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


def read_raw_jobs(spark: SparkSession, raw_dir: Path) -> DataFrame:
    """Read every bronze JSON document under ``raw_dir`` in one Spark scan.

    Each file is a single pretty-printed object with a ``jobs`` array.
    ``recursiveFileLookup`` picks up every ``*.json`` file, including a
    merged multi-source extract and older single-source landings.
    """
    if not raw_dir.is_dir():
        raise RawDataNotFoundError(f"Bronze directory does not exist: {raw_dir}")
    json_files = [path for path in raw_dir.glob("*.json") if path.is_file()]
    if not json_files:
        raise RawDataNotFoundError(
            f"No files matching {RAW_FILE_PATTERN} in {raw_dir}"
        )
    logger.info("Reading %s bronze file(s) from %s", len(json_files), raw_dir)
    raw = (
        spark.read.option("multiLine", "true")
        .option("recursiveFileLookup", "true")
        .json(raw_dir.as_posix())
    )
    if "jobs" not in raw.columns:
        raise JobTransformError(f"{raw_dir} does not contain a jobs array")
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
    # Offsets and fractional seconds make a strict pattern throw in ANSI mode.
    normalized = F.regexp_replace(text, r"(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})$", "")
    return F.coalesce(
        F.try_to_timestamp(normalized, F.lit("yyyy-MM-dd'T'HH:mm:ss")),
        F.try_to_timestamp(normalized, F.lit("yyyy-MM-dd HH:mm:ss")),
        F.try_to_timestamp(normalized, F.lit("yyyy-MM-dd")),
        F.try_to_timestamp(text),
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
    return _drop_duplicate_vacancies(relevant)


def _drop_duplicate_vacancies(jobs: DataFrame) -> DataFrame:
    """Keep one row per job id and per title/company pair.

    The newest ``publication_date`` wins. ``dropDuplicates`` then enforces
    both keys. Spark does not preserve order inside ``dropDuplicates``, so
    the row number runs first.
    """
    keyed = jobs.withColumn(
        "_dedupe_id",
        F.concat(F.lit("id:"), F.col("id").cast("string")),
    )
    newest = (
        keyed.withColumn(
            "_rank",
            F.row_number().over(
                Window.partitionBy("_dedupe_id").orderBy(
                    F.col("publication_date").desc_nulls_last()
                )
            ),
        )
        .where(F.col("_rank") == 1)
        .drop("_rank")
        .dropDuplicates(["_dedupe_id"])
        .dropDuplicates(["title", "company_name"])
        .drop("_dedupe_id")
    )
    return newest.select(
        "id",
        "title",
        "company_name",
        "category",
        "publication_date",
        "candidate_required_location",
        "salary",
        "description",
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


def _storage_target(root: str, relative_path: str) -> str:
    """Join a Delta root and a table path without breaking ``abfss://``."""
    return f"{root.rstrip('/')}/{relative_path.strip('/')}"


def write_gold_sample(
    fact: DataFrame,
    company: DataFrame,
    location: DataFrame,
    destination: Path = GOLD_SAMPLE_PATH,
) -> None:
    """Write ``fct_vacancies`` as one Snappy Parquet file for the dashboard.

    Company and location names are included so Streamlit Cloud can chart the
    fact grain without the rest of the Delta lake.
    """
    serving = (
        fact.join(company, "company_id", "left")
        .join(location, "location_id", "left")
        .select(
            "vacancy_id",
            "company_id",
            "location_id",
            "publication_date",
            "publication_timestamp",
            "title",
            "company_name",
            "category",
            "location_name",
            "salary",
            "description",
        )
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / "_gold_sample_staging"
    if staging.exists():
        shutil.rmtree(staging)
    (
        serving.coalesce(1)
        .write.mode("overwrite")
        .option("compression", "snappy")
        .parquet(staging.as_posix())
    )
    parts = [path for path in staging.glob("*.parquet") if path.is_file()]
    if len(parts) != 1:
        raise JobTransformError(
            f"Expected one Parquet part in {staging}, found {len(parts)}"
        )
    if destination.exists():
        destination.unlink()
    shutil.move(str(parts[0]), str(destination))
    shutil.rmtree(staging, ignore_errors=True)
    logger.info("Wrote compressed gold sample to %s", destination)


def write_table(frame: DataFrame, destination: str | Path) -> None:
    """Overwrite one Delta table at ``destination``."""
    target = destination.as_posix() if isinstance(destination, Path) else destination
    (
        frame.write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(target)
    )


def run_transformation(
    spark: SparkSession,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_root: str | None = None,
) -> dict[str, int]:
    """Read every bronze JSON file and write silver plus gold Delta tables.

    Returns a mapping of output-relative path to row count.
    """
    root = output_root or resolve_delta_root()
    raw_jobs = read_raw_jobs(spark, raw_dir)
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
            destination = _storage_target(root, relative_path)
            logger.info(
                "Writing %s row(s) as delta to %s",
                row_count,
                destination,
            )
            write_table(materialized, destination)
            counts[relative_path] = row_count
            if materialized is not silver:
                materialized.unpersist()
        write_gold_sample(
            gold["fct_vacancies"],
            gold["dim_company"],
            gold["dim_location"],
        )
    finally:
        silver.unpersist()

    if counts.get("silver/jobs", 0) == 0:
        logger.warning("Silver table is empty after the title filter")
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI for one silver/gold batch."""
    parser = argparse.ArgumentParser(
        description="Clean raw job JSON into silver and gold Delta tables."
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
        default=None,
        help=(
            "Delta root directory. Defaults to data/processed/delta, "
            "or an abfss:// path when STORAGE_TYPE=azure."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one transformation. Returns a process exit code."""
    configure_logging()
    args = parse_args(argv)
    try:
        output_root = (
            args.output_dir.as_posix()
            if args.output_dir is not None
            else resolve_delta_root()
        )
        spark = create_spark_session()
        try:
            run_transformation(
                spark,
                raw_dir=args.raw_dir,
                output_root=output_root,
            )
        finally:
            spark.stop()
    except JobTransformError:
        logger.exception("Job transformation failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
