"""Interactive view of the gold vacancy star schema.

Run from the repository root:

    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from pyspark.sql import SparkSession

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transform.clean_jobs import (  # noqa: E402
    create_spark_session,
    resolve_delta_root,
)

PAGE_TITLE = "Tech Job Market"
ACCENT = "#0F6E6B"
PALETTE = ["#0F6E6B", "#1F8A84", "#3D9B8F", "#7FB9A8", "#C4DDD4", "#E7F2EF"]


def _table_path(root: str, relative_path: str) -> str:
    """Join a Delta root and a table path without breaking ``abfss://``."""
    return f"{root.rstrip('/')}/{relative_path.strip('/')}"


@st.cache_resource(show_spinner=False)
def get_spark() -> SparkSession:
    """Reuse one local Spark session for the lifetime of the dashboard."""
    return create_spark_session("tech-job-dashboard")


@st.cache_data(show_spinner=False)
def load_vacancies() -> pd.DataFrame:
    """Join the gold fact table to company and location, then return pandas."""
    spark = get_spark()
    root = resolve_delta_root()
    fact = spark.read.format("delta").load(_table_path(root, "gold/fct_vacancies"))
    company = spark.read.format("delta").load(_table_path(root, "gold/dim_company"))
    location = spark.read.format("delta").load(_table_path(root, "gold/dim_location"))
    vacancies = (
        fact.join(company, "company_id", "left")
        .join(location, "location_id", "left")
        .select(
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
        .toPandas()
    )
    if vacancies.empty:
        return vacancies

    vacancies["publication_timestamp"] = pd.to_datetime(
        vacancies["publication_timestamp"], errors="coerce"
    )
    vacancies["publication_date"] = pd.to_datetime(
        vacancies["publication_date"], errors="coerce"
    ).dt.date
    for column in ("title", "company_name", "category", "location_name", "salary"):
        vacancies[column] = vacancies[column].fillna("").astype(str).str.strip()
    vacancies["description"] = vacancies["description"].fillna("").astype(str)
    return vacancies.sort_values(
        "publication_timestamp", ascending=False, na_position="last"
    )


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


def main() -> None:
    """Render the gold-layer dashboard."""
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=":bar_chart:",
        layout="wide",
    )
    st.title("Tech Job Market")
    st.caption(
        "Gold-laag van het Delta Lake-sterschema. "
        "Bron: Remotive, opgeschoond in de silver-laag."
    )

    try:
        with st.spinner("Delta-tabellen laden..."):
            vacancies = load_vacancies()
    except Exception as exc:
        st.error(
            "De gold-tabellen zijn nog niet leesbaar. "
            "Draai eerst de transformatie vanaf de projectroot."
        )
        st.code("python -m src.transform.clean_jobs", language="powershell")
        with st.expander("Technische details"):
            st.write(str(exc))
        st.stop()

    if vacancies.empty:
        st.info("De gold-laag bevat nog geen vacatures.")
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
