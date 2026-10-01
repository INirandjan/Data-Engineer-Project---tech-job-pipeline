"""Standalone Streamlit dashboard. No imports from this repository.

Run from the repository root:

    streamlit run src/dashboard/app.py

Streamlit Cloud uses the same file. The module loads with only installed
packages, so a missing ``src`` package on ``sys.path`` cannot raise
``ModuleNotFoundError``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time

import duckdb
import pandas as pd
import requests
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if not os.path.isdir(os.path.join(PROJECT_ROOT, "data")):
    PROJECT_ROOT = os.getcwd()

DELTA_GOLD_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "delta", "gold")
PARQUET_GOLD_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "gold")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
SAMPLE_PATH = os.path.join(PROJECT_ROOT, "data", "sample", "jobs_sample.json")
GOLD_SAMPLE_PATH = os.path.join(PROJECT_ROOT, "data", "sample", "gold_jobs_sample.parquet")
GOLD_SOURCE_LABEL = "Data Source: Gold Delta Layer (700+ Jobs processed via PySpark)"
GOLD_TABLES = ("fct_vacancies", "dim_company", "dim_location")

API_URL = "https://remotive.com/api/remote-jobs"
JOBICY_API_URL = "https://jobicy.com/api/v2/remote-jobs"
ARBEITNOW_API_URL = "https://www.arbeitnow.com/api/job-board-api"
SEARCH_TERM = "Data Engineer"
LIVE_CATEGORIES = ("dev", "data", "software-dev", "backend")
LIVE_ARBEITNOW_PAGES = 8
TITLE_KEYWORDS = ("data", "engineer", "developer", "python")
UNKNOWN_COMPANY = "Unknown"
UNSPECIFIED_LOCATION = "Unspecified"
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
HTML_ENTITIES = (
    ("&nbsp;", " "),
    ("&lt;", "<"),
    ("&gt;", ">"),
    ("&quot;", '"'),
    ("&#39;", "'"),
    ("&amp;", "&"),
)
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
SOURCE_GOLD_SAMPLE = "gold-sample"
SOURCE_PARQUET = "gold-parquet"
SOURCE_RAW = "bronze-json"
SOURCE_LIVE = "remotive-api"
SOURCE_SAMPLE = "sample"
SOURCE_EMPTY = "empty"


def _empty_vacancies() -> pd.DataFrame:
    """Return a vacancy frame with the columns the charts expect."""
    return pd.DataFrame(columns=list(VACANCY_COLUMNS))


def _as_path(path: str | os.PathLike[str]) -> str:
    """Return a filesystem path string."""
    return os.fspath(path)


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
    """Clean a Remotive ``jobs`` array into the dashboard vacancy grain."""
    rows: list[dict[str, object]] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        title = _clean_text(job.get("title"))
        if not _title_is_relevant(title):
            continue
        try:
            vacancy_id = int(job["id"])
        except (KeyError, TypeError, ValueError):
            continue
        published = pd.to_datetime(job.get("publication_date"), errors="coerce")
        company = _clean_text(job.get("company_name")) or UNKNOWN_COMPANY
        location = (
            _clean_text(job.get("candidate_required_location")) or UNSPECIFIED_LOCATION
        )
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
        .drop_duplicates(["title", "company_name"], keep="first")
        .sort_values("publication_timestamp", ascending=False, na_position="last")
        .reset_index(drop=True)
    )


def _read_job_list(path: str | os.PathLike[str]) -> list[dict]:
    """Read the ``jobs`` array from a bronze-shaped JSON document."""
    with open(_as_path(path), encoding="utf-8") as handle:
        payload = json.load(handle)
    jobs = payload.get("jobs") if isinstance(payload, dict) else payload
    if not isinstance(jobs, list):
        raise ValueError(f"{os.path.basename(_as_path(path))} does not contain a jobs array")
    return [job for job in jobs if isinstance(job, dict)]


def _raw_json_files(raw_dir: str | os.PathLike[str]) -> list[str]:
    """Return every bronze JSON file, oldest first."""
    directory = _as_path(raw_dir)
    if not os.path.isdir(directory):
        return []
    names = sorted(
        name
        for name in os.listdir(directory)
        if name.endswith(".json") and os.path.isfile(os.path.join(directory, name))
    )
    return [os.path.join(directory, name) for name in names]


def active_parquet_files(table_dir: str | os.PathLike[str]) -> list[str]:
    """Return the Parquet files that the current Delta snapshot still contains.

    A plain Parquet directory has no transaction log, so every ``*.parquet``
    file is current. A Delta table is replayed from ``_delta_log`` so removed
    files from older overwrites are not counted twice.
    """
    directory = _as_path(table_dir)
    if not os.path.isdir(directory):
        return []
    log_dir = os.path.join(directory, "_delta_log")
    commits = []
    if os.path.isdir(log_dir):
        commits = sorted(
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.endswith(".json") and os.path.isfile(os.path.join(log_dir, name))
        )
    if not commits:
        return sorted(
            os.path.join(directory, name).replace("\\", "/")
            for name in os.listdir(directory)
            if name.endswith(".parquet") and os.path.isfile(os.path.join(directory, name))
        )

    active: dict[str, None] = {}
    for commit in commits:
        with open(commit, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        for line in lines:
            if not line.strip():
                continue
            action = json.loads(line)
            if "add" in action and action["add"].get("path"):
                active[action["add"]["path"]] = None
            elif "remove" in action and action["remove"].get("path"):
                active.pop(action["remove"]["path"], None)
    files: list[str] = []
    for relative in active:
        path = os.path.join(directory, relative)
        if os.path.isfile(path):
            files.append(path.replace("\\", "/"))
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


def _prepare_vacancies(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize dtypes and sort the newest publication first."""
    if frame.empty:
        return _empty_vacancies()
    prepared = frame.copy()
    for column in VACANCY_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    prepared = prepared.loc[:, list(VACANCY_COLUMNS)]
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


def _read_gold_directory(gold_dir: str | os.PathLike[str]) -> pd.DataFrame | None:
    """Join fact, company, and location when all three tables have files."""
    root = _as_path(gold_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name in GOLD_TABLES:
        files = active_parquet_files(os.path.join(root, name))
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


def _publication(value: object) -> str:
    """Turn a unix timestamp or an API date string into text pandas can parse."""
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:
            seconds /= 1000
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
    return "" if value is None else str(value)


def _canonical_id(source: str, native_id: object) -> int:
    """Stable positive id so the same listing from one board stays unique."""
    digest = hashlib.sha256(f"{source}:{native_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def _http_json(url: str, params: dict[str, str]) -> dict:
    """GET one public job-board document. Callers catch failures."""
    response = requests.get(
        url,
        params=params,
        timeout=20,
        headers={
            "Accept": "application/json",
            "User-Agent": "tech-job-pipeline/0.1 (educational data-engineering project)",
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return payload


def _fetch_live_jobs() -> list[dict]:
    """Pull Remotive, Jobicy, and the first Arbeitnow pages without PySpark."""
    jobs: list[dict] = []
    for category in LIVE_CATEGORIES:
        try:
            payload = _http_json(
                os.environ.get("REMOTIVE_API_URL", API_URL),
                {"category": category, "search": category},
            )
            for job in payload.get("jobs") or []:
                if isinstance(job, dict) and job.get("id") is not None and job.get("title"):
                    copied = dict(job)
                    copied["id"] = _canonical_id("remotive", job.get("id"))
                    jobs.append(copied)
        except Exception:
            continue
        query = {"count": "50", "tag": category}
        if category == "dev":
            query = {"count": "50", "industry": "dev"}
        elif category == "data":
            query = {"count": "50", "tag": "data"}
        try:
            payload = _http_json(os.environ.get("JOBICY_API_URL", JOBICY_API_URL), query)
            for job in payload.get("jobs") or []:
                if not isinstance(job, dict) or job.get("id") is None:
                    continue
                title = job.get("jobTitle") or job.get("title")
                if not title:
                    continue
                jobs.append(
                    {
                        "id": _canonical_id("jobicy", job.get("id")),
                        "title": title,
                        "company_name": job.get("companyName") or job.get("company_name") or "",
                        "category": job.get("jobIndustry") or category,
                        "publication_date": _publication(job.get("pubDate")),
                        "candidate_required_location": job.get("jobGeo") or "",
                        "salary": job.get("salary") or "",
                        "description": job.get("jobDescription") or "",
                    }
                )
        except Exception:
            continue

    for page in range(1, LIVE_ARBEITNOW_PAGES + 1):
        try:
            payload = _http_json(
                os.environ.get("ARBEITNOW_API_URL", ARBEITNOW_API_URL),
                {"page": str(page)},
            )
        except Exception:
            break
        batch = payload.get("data") if isinstance(payload.get("data"), list) else []
        if not batch:
            break
        for job in batch:
            if not isinstance(job, dict) or not job.get("title"):
                continue
            native_id = job.get("slug") or job.get("url") or job.get("title")
            tags = job.get("tags") if isinstance(job.get("tags"), list) else []
            jobs.append(
                {
                    "id": _canonical_id("arbeitnow", native_id),
                    "title": job.get("title"),
                    "company_name": job.get("company_name") or "",
                    "category": str(tags[0]) if tags else "dev",
                    "publication_date": _publication(job.get("created_at")),
                    "candidate_required_location": job.get("location") or "",
                    "salary": "",
                    "description": job.get("description") or "",
                }
            )
        links = payload.get("links") if isinstance(payload.get("links"), dict) else {}
        if not links.get("next"):
            break
    if not jobs:
        raise ValueError("Remotive, Jobicy, and Arbeitnow returned no listings")
    return jobs


def _save_raw(payload: dict, raw_dir: str | os.PathLike[str]) -> None:
    """Write a bronze file when the process can create ``data/raw``."""
    directory = _as_path(raw_dir)
    try:
        os.makedirs(directory, exist_ok=True)
        name = time.strftime("jobs_raw_%Y%m%d_%H%M%S.json", time.gmtime())
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError:
        return


def _read_gold_sample(path: str | os.PathLike[str]) -> pd.DataFrame | None:
    """Read the single compressed fact sample shipped with the repository."""
    sample = _as_path(path)
    if not os.path.isfile(sample):
        return None
    quoted = sample.replace("\\", "/").replace("'", "''")
    connection = duckdb.connect(database=":memory:")
    try:
        frame = connection.execute(f"SELECT * FROM read_parquet('{quoted}')").df()
    finally:
        connection.close()
    return _prepare_vacancies(frame)


def resolve_vacancies(
    delta_root: str | os.PathLike[str] | None = None,
    parquet_root: str | os.PathLike[str] | None = None,
    raw_dir: str | os.PathLike[str] | None = None,
    sample_path: str | os.PathLike[str] | None = None,
    gold_sample_path: str | os.PathLike[str] | None = None,
    allow_fetch: bool = True,
) -> tuple[pd.DataFrame, str]:
    """Load vacancies from Delta, the shipped Parquet sample, or a later fallback.

    Every missing path and every failed read is skipped. Streamlit Cloud has
    no ``data/processed`` tree, so the compressed fact sample is the next source.
    """
    delta_root = DELTA_GOLD_DIR if delta_root is None else delta_root
    parquet_root = PARQUET_GOLD_DIR if parquet_root is None else parquet_root
    raw_dir = RAW_DIR if raw_dir is None else raw_dir
    sample_path = SAMPLE_PATH if sample_path is None else sample_path
    gold_sample_path = GOLD_SAMPLE_PATH if gold_sample_path is None else gold_sample_path

    try:
        frame = _read_gold_directory(delta_root)
    except Exception:
        frame = None
    if frame is not None and not frame.empty:
        return frame, SOURCE_GOLD

    try:
        frame = _read_gold_sample(gold_sample_path)
    except Exception:
        frame = None
    if frame is not None and not frame.empty:
        return frame, SOURCE_GOLD_SAMPLE

    try:
        frame = _read_gold_directory(parquet_root)
    except Exception:
        frame = None
    if frame is not None and not frame.empty:
        return frame, SOURCE_PARQUET

    try:
        raw_jobs: list[dict] = []
        for raw_file in _raw_json_files(raw_dir):
            raw_jobs.extend(_read_job_list(raw_file))
        if raw_jobs:
            frame = vacancies_from_jobs(raw_jobs)
            if not frame.empty:
                return frame, SOURCE_RAW
    except Exception:
        pass

    if allow_fetch:
        try:
            jobs = _fetch_live_jobs()
            frame = vacancies_from_jobs(jobs)
            if not frame.empty:
                _save_raw({"job-count": len(jobs), "jobs": jobs}, raw_dir)
                return frame, SOURCE_LIVE
        except Exception:
            pass

    sample = _as_path(sample_path)
    if os.path.isfile(sample):
        try:
            frame = vacancies_from_jobs(_read_job_list(sample))
            if not frame.empty:
                return frame, SOURCE_SAMPLE
        except Exception:
            pass
    return _empty_vacancies(), SOURCE_EMPTY


def _directory_token(root: str | os.PathLike[str]) -> str:
    """Change when gold files are rewritten, so the Streamlit cache drops."""
    directory = _as_path(root)
    if not os.path.isdir(directory):
        return ""
    stamps: list[int] = []
    for dirpath, _dirnames, filenames in os.walk(directory):
        for name in filenames:
            if name.endswith(".parquet") or name.endswith(".json"):
                try:
                    stamps.append(os.stat(os.path.join(dirpath, name)).st_mtime_ns)
                except OSError:
                    continue
    if not stamps:
        return ""
    return f"{len(stamps)}:{max(stamps)}"


@st.cache_data(ttl=6 * 60 * 60, show_spinner=False)
def load_vacancies(gold_token: str, raw_token: str) -> tuple[pd.DataFrame, str]:
    """Cache the resolved vacancy frame for one Streamlit session."""
    del gold_token, raw_token
    return resolve_vacancies()


def _count_by(frame: pd.DataFrame, column: str, limit: int) -> pd.DataFrame:
    """Aggregate one column with DuckDB. Failures become an empty chart frame."""
    if column not in {"category", "location_name"} or frame.empty:
        return pd.DataFrame(columns=["label", "aantal"])
    try:
        duckdb.register("vacancies", frame)
        return duckdb.query(
            f"""
            SELECT {column} AS label, COUNT(*)::BIGINT AS aantal
            FROM vacancies
            WHERE {column} IS NOT NULL AND TRIM(CAST({column} AS VARCHAR)) <> ''
            GROUP BY 1
            ORDER BY aantal DESC
            LIMIT {int(limit)}
            """
        ).df()
    except Exception:
        return pd.DataFrame(columns=["label", "aantal"])


def _matches(frame: pd.DataFrame, query: str) -> pd.DataFrame:
    """Keep rows whose visible text contains ``query``."""
    if frame.empty or not query.strip():
        return frame
    haystack = (
        frame["title"].fillna("").str.cat(frame["company_name"].fillna(""), sep=" ")
        .str.cat(frame["category"].fillna(""), sep=" ")
        .str.cat(frame["location_name"].fillna(""), sep=" ")
        .str.cat(frame["description"].fillna(""), sep=" ")
        .str.casefold()
    )
    return frame.loc[haystack.str.contains(query.casefold(), regex=False)].copy()


def _source_caption(source: str) -> str:
    """Dutch caption for the source that actually supplied the rows."""
    captions = {
        SOURCE_GOLD: GOLD_SOURCE_LABEL,
        SOURCE_GOLD_SAMPLE: GOLD_SOURCE_LABEL,
        SOURCE_PARQUET: (
            "Gold Parquet-bestanden, gelezen met DuckDB. Bron: Remotive."
        ),
        SOURCE_RAW: "Nieuwste bronze JSON, in Python opgeschoond. Bron: Remotive.",
        SOURCE_LIVE: (
            "Live opgehaald bij Remotive, Jobicy en Arbeitnow, daarna in Python opgeschoond."
        ),
        SOURCE_SAMPLE: (
            "Demo-set uit data/sample, omdat er nog geen gold- of bronze-data is."
        ),
        SOURCE_EMPTY: "Geen vacatures beschikbaar.",
    }
    return captions.get(source, "Bron: Remotive.")


def _bar(counts: pd.DataFrame, title: str) -> None:
    """Render one aggregated bar chart, or a short note when it is empty."""
    st.subheader(title)
    if counts.empty:
        st.info("Geen waarden om te tonen.")
        return
    try:
        st.bar_chart(counts.set_index("label")["aantal"])
    except Exception:
        st.dataframe(counts, use_container_width=True, hide_index=True)


def main() -> None:
    """Render the vacancy dashboard."""
    st.set_page_config(
        page_title="Tech Job Market",
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Tech Job Market")

    try:
        with st.spinner("Vacatures laden..."):
            sample_token = ""
            if os.path.isfile(GOLD_SAMPLE_PATH):
                sample_token = str(os.stat(GOLD_SAMPLE_PATH).st_mtime_ns)
            vacancies, source = load_vacancies(
                _directory_token(DELTA_GOLD_DIR) + _directory_token(PARQUET_GOLD_DIR) + sample_token,
                _directory_token(RAW_DIR),
            )
    except Exception as exc:
        vacancies, source = _empty_vacancies(), SOURCE_EMPTY
        st.error("De vacatures konden niet worden geladen.")
        with st.expander("Technische details"):
            st.write(str(exc))

    if source in {SOURCE_GOLD, SOURCE_GOLD_SAMPLE}:
        st.markdown(
            '<p style="display:inline-block;margin:0 0 0.75rem;padding:0.2rem 0.7rem;'
            'border-radius:999px;background:#E7F2EF;color:#0F6E6B;font-size:0.85rem;">'
            f"{GOLD_SOURCE_LABEL}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.caption(_source_caption(source))
    if source == SOURCE_SAMPLE:
        st.info(
            "Er staan nog geen Gold-tabellen of bronze-bestanden op deze server. "
            "Het dashboard toont de gebundelde demo."
        )
    if vacancies.empty:
        st.info("Er zijn nog geen vacatures om te tonen.")
        return

    companies = vacancies["company_name"].replace("", pd.NA).nunique(dropna=True)
    locations = vacancies["location_name"].replace("", pd.NA).nunique(dropna=True)
    metric_vacancies, metric_companies, metric_locations = st.columns(3)
    metric_vacancies.metric("Totaal aantal verwerkte vacatures", f"{len(vacancies)}")
    metric_companies.metric("Unieke bedrijven", f"{companies}")
    metric_locations.metric("Locaties", f"{locations}")

    chart_roles, chart_locations = st.columns(2)
    with chart_roles:
        _bar(_count_by(vacancies, "category", limit=5), "Top 5 meest gevraagde categorieën")
    with chart_locations:
        _bar(_count_by(vacancies, "location_name", limit=8), "Verdeling van locaties")

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
