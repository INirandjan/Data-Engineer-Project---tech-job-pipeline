"""Interactive view of cleaned tech vacancies, without a JVM.

Run from the repository root:

    streamlit run src/dashboard/app.py

Streamlit Cloud can use the same entry point. The app never imports PySpark.
It reads local Delta or Parquet with DuckDB when those files exist. Otherwise
it transforms the newest bronze JSON, fetches the Remotive API, or loads
``data/sample/jobs_sample.json``.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from src.extract.fetch_jobs import (
    DEFAULT_API_URL,
    DEFAULT_SEARCH_TERM,
    JobExtractionError,
    fetch_jobs,
    load_settings,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELTA_GOLD_DIR = PROJECT_ROOT / "data" / "processed" / "delta" / "gold"
PARQUET_GOLD_DIR = PROJECT_ROOT / "data" / "processed" / "gold"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SAMPLE_PATH = PROJECT_ROOT / "data" / "sample" / "jobs_sample.json"
RAW_FILE_PATTERN = "jobs_raw_*.json"
GOLD_TABLES = ("fct_vacancies", "dim_company", "dim_location")

TITLE_KEYWORDS: tuple[str, ...] = ("data", "engineer", "developer", "python")
UNKNOWN_COMPANY = "Unknown"
UNSPECIFIED_LOCATION = "Unspecified"
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ENTITIES: tuple[tuple[str, str], ...] = (
    ("&nbsp;", " "),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&amp;", "&"),
)

PAGE_TITLE = "Tech Job Market"
ACCENT = "#0F6E6B"
PALETTE = ["#0F6E6B", "#1F8A84", "#3D9B8F", "#7FB9A8", "#C4DDD4", "#E7F2EF"]
VACANCY_COLUMNS = (
    "vacancy_id",
    "publication_date",
    "publication_timestamp",
    "title",
    "company_name",
    "category",
    "location_name",
    "salary",
    "description",
)

SOURCE_GOLD = "gold-delta"
SOURCE_PARQUET = "gold-parquet"
SOURCE_RAW = "bronze-json"
SOURCE_LIVE = "remotive-api"
SOURCE_SAMPLE = "sample"


def _empty_vacancies() -> pd.DataFrame:
    """Return a vacancy frame with the columns the charts expect."""
    return pd.DataFrame(columns=list(VACANCY_COLUMNS))


def _clean_text(value: object) -> str | None:
    """Trim text and turn blanks into null."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _strip_html(value: object) -> str | None:
    """Remove HTML tags and common entities, then collapse whitespace."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = HTML_TAG_PATTERN.sub(" ", str(value))
    for entity, replacement in HTML_ENTITIES:
        text = text.replace(entity, replacement)
    return _clean_text(text)


def _title_is_relevant(title: str | None) -> bool:
    """True when the title contains one of the IT role keywords."""
    if not title:
        return False
    lowered = title.casefold()
    return any(keyword in lowered for keyword in TITLE_KEYWORDS)


def vacancies_from_jobs(jobs: list[dict]) -> pd.DataFrame:
    """Clean a Remotive ``jobs`` array into the dashboard vacancy grain.

    This is the lightweight stand-in for the Spark silver/gold job. It keeps
    the same title filter, HTML cleanup, and unknown-company rules, and
    returns one flat row per vacancy so the app does not need the JVM.
    """
    rows: list[dict[str, object]] = []
    for job in jobs:
        title = _clean_text(job.get("title"))
        if not _title_is_relevant(title):
            continue
        try:
            vacancy_id = int(job["id"])
        except (KeyError, TypeError, ValueError):
            continue
        published = pd.to_datetime(job.get("publication_date"), errors="coerce")
        company = _clean_text(job.get("company_name")) or UNKNOWN_COMPANY
        location = _clean_text(job.get("candidate_required_location")) or UNSPECIFIED_LOCATION
        rows.append(
            {
                "vacancy_id": vacancy_id,
                "publication_date": published.date() if pd.notna(published) else None,
                "publication_timestamp": published,
                "title": title,
                "company_name": company,
                "category": _clean_text(job.get("category")) or "",
                "location_name": location,
                "salary": _clean_text(job.get("salary")) or "",
                "description": _strip_html(job.get("description")) or "",
            }
        )
    frame = pd.DataFrame(rows, columns=list(VACANCY_COLUMNS))
    if frame.empty:
        return frame
    frame = frame.sort_values(
        "publication_timestamp", ascending=False, na_position="last"
    )
    return (
        frame.drop_duplicates("vacancy_id", keep="first")
        .sort_values("publication_timestamp", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _read_job_list(path: Path) -> list[dict]:
    """Read the ``jobs`` array from a bronze-shaped JSON document."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(jobs, list):
        raise ValueError(f"{path.name} does not contain a jobs array")
    return [job for job in jobs if isinstance(job, dict)]


def _latest_raw_file(raw_dir: Path) -> Path | None:
    """Return the newest ``jobs_raw_<timestamp>.json`` file, if any."""
    if not raw_dir.is_dir():
        return None
    candidates = sorted(path for path in raw_dir.glob(RAW_FILE_PATTERN) if path.is_file())
    return candidates[-1] if candidates else None


def active_parquet_files(table_dir: Path) -> list[str]:
    """Return the Parquet files that the current Delta snapshot still contains.

    A plain Parquet directory has no transaction log, so every ``*.parquet``
    file is current. A Delta table is replayed from ``_delta_log`` so removed
    files from older overwrites are not counted twice.
    """
    if not table_dir.is_dir():
        return []
    log_dir = table_dir / "_delta_log"
    commits = sorted(log_dir.glob("*.json")) if log_dir.is_dir() else []
    if not commits:
        return sorted(path.as_posix() for path in table_dir.glob("*.parquet") if path.is_file())

    active: dict[str, None] = {}
    for commit in commits:
        for line in commit.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            action = json.loads(line)
            if "add" in action and action["add"].get("path"):
                active[action["add"]["path"]] = None
            elif "remove" in action and action["remove"].get("path"):
                active.pop(action["remove"]["path"], None)
    files: list[str] = []
    for relative in active:
        path = table_dir / relative
        if path.is_file():
            files.append(path.as_posix())
    return files


def _read_parquet(paths: list[str]) -> pd.DataFrame:
    """Read one or more Parquet files with DuckDB."""
    if not paths:
        return pd.DataFrame()
    quoted = ", ".join("'" + path.replace("'", "''") + "'" for path in paths)
    connection = duckdb.connect(database=":memory:")
    try:
        return connection.execute(f"SELECT * FROM read_parquet([{quoted}])").df()
    finally:
        connection.close()


def _read_gold_directory(gold_dir: Path) -> pd.DataFrame | None:
    """Join fact, company, and location when all three tables have files."""
    tables: dict[str, pd.DataFrame] = {}
    for name in GOLD_TABLES:
        files = active_parquet_files(gold_dir / name)
        if not files:
            return None
        tables[name] = _read_parquet(files)
    connection = duckdb.connect(database=":memory:")
    try:
        connection.register("fact", tables["fct_vacancies"])
        connection.register("company", tables["dim_company"])
        connection.register("location", tables["dim_location"])
        frame = connection.execute(
            """
            SELECT
                fact.vacancy_id,
                fact.publication_date,
                fact.publication_timestamp,
                fact.title,
                company.company_name,
                fact.category,
                location.location_name,
                fact.salary,
                fact.description
            FROM fact
            LEFT JOIN company ON fact.company_id = company.company_id
            LEFT JOIN location ON fact.location_id = location.location_id
            """
        ).df()
    finally:
        connection.close()
    return _prepare_vacancies(frame)


def _prepare_vacancies(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and sort the newest publication first."""
    if frame.empty:
        return _empty_vacancies()
    prepared = frame.copy()
    prepared["publication_timestamp"] = pd.to_datetime(
        prepared["publication_timestamp"], errors="coerce"
    )
    prepared["publication_date"] = pd.to_datetime(
        prepared["publication_date"], errors="coerce"
    ).dt.date
    for column in ("title", "company_name", "category", "location_name", "salary"):
        prepared[column] = prepared[column].fillna("").astype(str).str.strip()
    prepared["description"] = prepared["description"].fillna("").astype(str)
    return prepared.sort_values(
        "publication_timestamp", ascending=False, na_position="last"
    ).reset_index(drop=True)


def resolve_vacancies(
    delta_root: Path | None = None,
    parquet_root: Path | None = None,
    raw_dir: Path | None = None,
    sample_path: Path | None = None,
    allow_fetch: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load vacancies from gold files, bronze JSON, the live API, or the sample.

    The first source that yields rows wins. ``allow_fetch`` is off in tests
    so a missing lake does not call Remotive.
    """
    delta_root = DELTA_GOLD_DIR if delta_root is None else delta_root
    parquet_root = PARQUET_GOLD_DIR if parquet_root is None else parquet_root
    raw_dir = RAW_DIR if raw_dir is None else raw_dir
    sample_path = SAMPLE_PATH if sample_path is None else sample_path

    for root, source in ((delta_root, SOURCE_GOLD), (parquet_root, SOURCE_PARQUET)):
        try:
            frame = _read_gold_directory(root)
        except (OSError, ValueError, duckdb.Error, json.JSONDecodeError) as exc:
            logger.warning("Could not read gold tables in %s: %s", root, exc)
            frame = None
        if frame is not None and not frame.empty:
            return frame, source

    raw_file = _latest_raw_file(raw_dir)
    if raw_file is not None:
        frame = vacancies_from_jobs(_read_job_list(raw_file))
        if not frame.empty:
            return frame, SOURCE_RAW

    if allow_fetch:
        try:
            settings = load_settings()
            payload = fetch_jobs(
                settings.get("search_term") or DEFAULT_SEARCH_TERM,
                api_url=settings.get("api_url") or DEFAULT_API_URL,
            )
            frame = vacancies_from_jobs(payload.get("jobs") or [])
            if not frame.empty:
                return frame, SOURCE_LIVE
        except (JobExtractionError, OSError, ValueError) as exc:
            logger.warning("Live Remotive fetch failed, using the sample: %s", exc)

    if not sample_path.is_file():
        raise FileNotFoundError(
            f"No gold data, bronze JSON, or sample file at {sample_path}"
        )
    frame = vacancies_from_jobs(_read_job_list(sample_path))
    return frame, SOURCE_SAMPLE


def _directory_token(root: Path) -> str:
    """Change when gold files are rewritten, so the Streamlit cache drops."""
    if not root.is_dir():
        return ""
    stamps: list[int] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix in {".parquet", ".json"}:
            stamps.append(path.stat().st_mtime_ns)
    if not stamps:
        return ""
    return f"{len(stamps)}:{max(stamps)}"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_vacancies(gold_token: str, raw_token: str) -> tuple[pd.DataFrame, str]:
    """Cache the resolved vacancy frame for one Streamlit session.

    ``gold_token`` and ``raw_token`` are cache keys. A new Delta commit or
    bronze file misses the cache immediately. A live API response stays
    cached for six hours, within Remotive's request guidance.
    """
    del gold_token, raw_token
    return resolve_vacancies()


def _top_counts(frame: pd.DataFrame, column: str, limit: int = 5) -> pd.DataFrame:
    """Return the most frequent non-blank values in ``column``."""
    values = frame[column].replace("", pd.NA).dropna()
    counts = values.value_counts().head(limit).rename_axis(column).reset_index(name="aantal")
    return counts


def _matches(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    """Keep rows whose visible text contains ``query``."""
    if not query.strip():
        return frame
    haystack = (
        frame["title"].str.cat(frame["company_name"], sep=" ")
        .str.cat(frame["category"], sep=" ")
        .str.cat(frame["location_name"], sep=" ")
        .str.cat(frame["description"], sep=" ")
        .str.casefold()
    )
    return frame.loc[haystack.str.contains(query.casefold(), regex=False)].copy()


def _bar_chart(counts: pd.DataFrame, label: str, title: str):
    """Horizontal bar chart, largest value at the top."""
    figure = px.bar(
        counts.sort_values("aantal", ascending=True),
        x="aantal",
        y=label,
        orientation="h",
        text="aantal",
        color_discrete_sequence=[ACCENT],
    )
    figure.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        height=380,
        xaxis_title="Aantal vacatures",
        yaxis_title="",
    )
    figure.update_traces(textposition="outside", cliponaxis=False)
    return figure


def _location_chart(counts: pd.DataFrame):
    """Donut chart of the published location requirements."""
    figure = px.pie(
        counts,
        names="location_name",
        values="aantal",
        hole=0.46,
        color_discrete_sequence=PALETTE,
    )
    figure.update_layout(
        title="Verdeling van locaties",
        template="plotly_white",
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        height=380,
        legend_title_text="",
    )
    figure.update_traces(textposition="inside", textinfo="percent")
    return figure


def _source_caption(source: str) -> str:
    """Dutch caption for the source that actually supplied the rows."""
    captions = {
        SOURCE_GOLD: (
            "Gold Delta-tabellen, gelezen met DuckDB. "
            "Bron: Remotive, opgeschoond in de silver-laag."
        ),
        SOURCE_PARQUET: (
            "Gold Parquet-bestanden, gelezen met DuckDB. "
            "Bron: Remotive, opgeschoond in de silver-laag."
        ),
        SOURCE_RAW: (
            "Nieuwste bronze JSON, in Python opgeschoond. "
            "Bron: Remotive."
        ),
        SOURCE_LIVE: (
            "Live opgehaald bij de Remotive API en in Python opgeschoond."
        ),
        SOURCE_SAMPLE: (
            "Demo-set uit data/sample, omdat er nog geen gold- of bronze-data is."
        ),
    }
    return captions.get(source, "Bron: Remotive.")


def main() -> None:
    """Render the vacancy dashboard."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Tech Job Market")

    try:
        with st.spinner("Vacatures laden..."):
            vacancies, source = load_vacancies(
                _directory_token(DELTA_GOLD_DIR) + _directory_token(PARQUET_GOLD_DIR),
                _directory_token(RAW_DIR),
            )
    except Exception as exc:
        st.error("De vacatures konden niet worden geladen.")
        with st.expander("Technische details"):
            st.write(str(exc))
        st.stop()

    st.caption(_source_caption(source))
    if source == SOURCE_SAMPLE:
        st.info(
            "Er staan nog geen Gold-tabellen of bronze-bestanden op deze server. "
            "Het dashboard toont de gebundelde demo."
        )

    if vacancies.empty:
        st.info("Er zijn nog geen vacatures om te tonen.")
        st.stop()

    companies = vacancies["company_name"].replace("", pd.NA).nunique(dropna=True)
    locations = vacancies["location_name"].replace("", pd.NA).nunique(dropna=True)
    metric_vacancies, metric_companies, metric_locations = st.columns(3)
    metric_vacancies.metric("Totaal aantal verwerkte vacatures", f"{len(vacancies)}")
    metric_companies.metric("Bedrijven", f"{companies}")
    metric_locations.metric("Locaties", f"{locations}")

    categories = _top_counts(vacancies, "category", limit=5)
    location_counts = _top_counts(vacancies, "location_name", limit=8)
    chart_roles, chart_locations = st.columns(2)
    with chart_roles:
        if categories.empty:
            st.info("Geen categorieën om te tonen.")
        else:
            st.plotly_chart(
                _bar_chart(categories, "category", "Top 5 meest gevraagde categorieën"),
                use_container_width=True,
            )
    with chart_locations:
        if location_counts.empty:
            st.info("Geen locaties om te tonen.")
        else:
            st.plotly_chart(
                _location_chart(location_counts),
                use_container_width=True,
            )

    st.subheader("Meest recente vacatures")
    query = st.text_input(
        "Zoek in titel, bedrijf, categorie, locatie of beschrijving",
        placeholder="Bijvoorbeeld Data Engineer of Python",
    )
    visible = _matches(vacancies, query)
    st.caption(f"{len(visible)} van {len(vacancies)} vacatures")
    display = visible.drop(columns=["publication_timestamp"]).rename(
        columns={
            "vacancy_id": "Id",
            "publication_date": "Publicatiedatum",
            "title": "Titel",
            "company_name": "Bedrijf",
            "category": "Categorie",
            "location_name": "Locatie",
            "salary": "Salaris",
            "description": "Beschrijving",
        }
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Beschrijving": st.column_config.TextColumn("Beschrijving", width="large"),
        },
    )


if __name__ == "__main__":
    main()
