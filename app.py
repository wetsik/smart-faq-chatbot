from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib import error, parse, request

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Lead Scoring & Pipeline Manager",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

HEALTH_SERVER_STARTED = False
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_LEADS_PATH = BASE_DIR / "sample_leads.csv"

NUMERIC_COLUMNS = [
    "budget",
    "deal_value",
    "last_activity_days",
    "meetings_booked",
    "email_open_rate",
    "employees",
]

TEXT_COLUMNS = [
    "lead_id",
    "company",
    "manager",
    "industry",
    "source",
    "region",
    "stage",
]

REQUIRED_COLUMNS = TEXT_COLUMNS + NUMERIC_COLUMNS
OPTIONAL_COLUMNS = ["bitrix_owner"]

STAGE_ORDER = ["New", "Contacted", "Qualified", "Proposal", "Negotiation", "Won", "Lost"]

CSS = """
<style>
:root {
    --bg-start: #111827;
    --bg-end: #0f172a;
    --surface: rgba(31, 41, 55, 0.92);
    --surface-strong: rgba(17, 24, 39, 0.96);
    --ink: #f9fafb;
    --muted: #cbd5e1;
    --teal: #5eead4;
    --gold: #fbbf24;
    --rose: #fb923c;
    --border: rgba(148, 163, 184, 0.18);
    --shadow: 0 18px 40px rgba(2, 6, 23, 0.35);
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(94,234,212,0.08), transparent 26%),
        radial-gradient(circle at top right, rgba(251,191,36,0.08), transparent 24%),
        linear-gradient(180deg, var(--bg-start) 0%, var(--bg-end) 100%);
}

.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 1.5rem;
    border-radius: 1.5rem;
    border: 1px solid var(--border);
    background:
        linear-gradient(135deg, rgba(94,234,212,0.08), rgba(251,191,36,0.07)),
        var(--surface-strong);
    box-shadow: var(--shadow);
}

.hero h1 {
    margin: 0;
    font-size: 2.2rem;
    line-height: 1.1;
    color: var(--ink);
}

.hero p {
    margin: 0.55rem 0 0;
    color: var(--muted);
    max-width: 70rem;
}

.hero-tags {
    margin-top: 1rem;
}

.hero-tag {
    display: inline-block;
    margin-right: 0.45rem;
    margin-bottom: 0.45rem;
    padding: 0.38rem 0.78rem;
    border-radius: 999px;
    background: rgba(94,234,212,0.12);
    color: #99f6e4;
    font-size: 0.84rem;
    font-weight: 600;
}

.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 1.2rem;
    padding: 1rem 1.1rem;
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.04);
    height: 100%;
    color: var(--ink);
}

.section-title {
    font-size: 1.02rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
    color: var(--ink);
}

.section-subtle {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 0.9rem;
}

.mini-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.65rem;
}

.mini-stat {
    background: rgba(15, 23, 42, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 1rem;
    padding: 0.75rem;
}

.mini-label {
    color: var(--muted);
    font-size: 0.82rem;
}

.mini-value {
    font-size: 1.2rem;
    font-weight: 700;
    margin-top: 0.15rem;
    color: var(--ink);
}

.priority-pill {
    display: inline-block;
    padding: 0.32rem 0.72rem;
    border-radius: 999px;
    font-size: 0.83rem;
    font-weight: 700;
}

.priority-high {
    background: rgba(16, 185, 129, 0.16);
    color: #6ee7b7;
}

.priority-medium {
    background: rgba(251, 191, 36, 0.16);
    color: #fcd34d;
}

.priority-low {
    background: rgba(251, 146, 60, 0.16);
    color: #fdba74;
}

.lead-strip {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: center;
    padding: 0.75rem 0;
    border-bottom: 1px dashed rgba(148, 163, 184, 0.14);
}

.lead-strip:last-child {
    border-bottom: none;
    padding-bottom: 0;
}

.lead-meta {
    color: var(--muted);
    font-size: 0.88rem;
}

.bar-track {
    width: 100%;
    height: 0.55rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #14b8a6 0%, #f59e0b 100%);
}

.insight-list {
    margin: 0;
    padding-left: 1.1rem;
    color: var(--ink);
}

.insight-list li {
    margin-bottom: 0.45rem;
}

.qa-board {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.85rem;
}

.qa-column {
    background: rgba(15, 23, 42, 0.35);
    border: 1px solid rgba(148, 163, 184, 0.16);
    border-radius: 1rem;
    padding: 0.85rem;
    min-height: 10rem;
}

.qa-title {
    color: var(--ink);
    font-weight: 800;
    margin-bottom: 0.7rem;
}

.qa-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 0.8rem;
    color: var(--ink);
    padding: 0.72rem;
    margin-bottom: 0.55rem;
    font-size: 0.9rem;
}

[data-testid="stMetricLabel"] {
    color: #cbd5e1;
}

[data-testid="stMetricValue"] {
    color: #f9fafb;
}

.stCaption {
    color: #cbd5e1;
}
</style>
"""


class HealthHandler(BaseHTTPRequestHandler):
    def _send_ok(self, body: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        if body:
            self.wfile.write(b"ok")

    def do_GET(self) -> None:
        if self.path in ("/", "/health"):
            self._send_ok(body=True)
            return
        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"not found")

    def do_HEAD(self) -> None:
        if self.path in ("/", "/health"):
            self._send_ok(body=False)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args) -> None:
        return


def start_health_server() -> None:
    global HEALTH_SERVER_STARTED

    if HEALTH_SERVER_STARTED:
        return

    enabled = os.environ.get("HEALTHCHECK_ENABLED", "true").lower() == "true"
    if not enabled:
        return

    health_port_raw = os.environ.get("HEALTHCHECK_PORT", "").strip()
    if not health_port_raw:
        return

    try:
        health_port = int(health_port_raw)
    except ValueError:
        return

    streamlit_port_raw = os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("PORT", "8501")
    try:
        streamlit_port = int(streamlit_port_raw)
    except ValueError:
        streamlit_port = 8501

    if health_port == streamlit_port:
        return

    def run_server() -> None:
        server = ThreadingHTTPServer(("0.0.0.0", health_port), HealthHandler)
        server.serve_forever()

    threading.Thread(target=run_server, daemon=True).start()
    HEALTH_SERVER_STARTED = True


def normalize_number(value, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return default


def init_state() -> None:
    st.session_state.setdefault(
        "loaded_leads",
        pd.DataFrame(
            columns=REQUIRED_COLUMNS
            + OPTIONAL_COLUMNS
            + ["score", "priority", "conversion_probability", "pricing_recommendation", "strategy_note"]
        ),
    )
    st.session_state.setdefault("response_seconds", None)
    st.session_state.setdefault("dataset_label", "No dataset loaded")
    st.session_state.setdefault("data_quality_board", {"Errors": [], "Warnings": [], "To Do": []})
    st.session_state.setdefault("support_tickets", [])


def get_config_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value

    try:
        return str(st.secrets.get(name, "")).strip()
    except (AttributeError, FileNotFoundError, KeyError):
        return ""


def create_trello_card(title: str, description: str, priority: str) -> tuple[bool, str]:
    api_key = get_config_value("TRELLO_API_KEY")
    token = get_config_value("TRELLO_TOKEN")
    list_id = get_config_value("TRELLO_LIST_ID")

    if not api_key or not token or not list_id:
        return False, "Trello API is not configured. Add TRELLO_API_KEY, TRELLO_TOKEN, and TRELLO_LIST_ID."

    card_description = f"Priority: {priority}\n\n{description}"
    payload = parse.urlencode(
        {
            "key": api_key,
            "token": token,
            "idList": list_id,
            "name": title,
            "desc": card_description,
        }
    ).encode("utf-8")

    trello_request = request.Request(
        "https://api.trello.com/1/cards",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    try:
        with request.urlopen(trello_request, timeout=12) as response:
            body = json.loads(response.read().decode("utf-8"))
            card_url = body.get("shortUrl") or body.get("url") or "Trello card created."
            return True, card_url
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        return False, f"Trello API error {exc.code}: {details or exc.reason}"
    except error.URLError as exc:
        return False, f"Trello connection error: {exc.reason}"
    except TimeoutError:
        return False, "Trello request timed out."


def is_trello_configured() -> bool:
    return all(
        [
            get_config_value("TRELLO_API_KEY"),
            get_config_value("TRELLO_TOKEN"),
            get_config_value("TRELLO_LIST_ID"),
        ]
    )


def load_leads_from_upload(uploaded_file) -> pd.DataFrame:
    decoded = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    rows = list(csv.DictReader(io.StringIO(decoded)))
    if not rows:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    return pd.DataFrame(rows)


def load_sample_leads() -> pd.DataFrame:
    if not SAMPLE_LEADS_PATH.exists():
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    return pd.read_csv(SAMPLE_LEADS_PATH)


def build_data_quality_board(df: pd.DataFrame) -> dict[str, list[str]]:
    board = {"Errors": [], "Warnings": [], "To Do": []}

    if df.empty:
        board["Errors"].append("CSV has no lead rows to analyze.")
        return board

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    for column in missing_columns:
        board["Errors"].append(f"Missing required column: {column}.")

    existing_required = [column for column in REQUIRED_COLUMNS if column in df.columns]
    for column in existing_required:
        empty_count = int(df[column].isna().sum() + (df[column].astype(str).str.strip() == "").sum())
        if empty_count:
            board["Errors"].append(f"{column} has {empty_count} empty value(s).")

    if "lead_id" in df.columns:
        duplicate_count = int(df["lead_id"].astype(str).str.strip().duplicated().sum())
        if duplicate_count:
            board["Warnings"].append(f"Found {duplicate_count} duplicate lead_id value(s).")

    for column in [column for column in NUMERIC_COLUMNS if column in df.columns]:
        raw_values = df[column].astype(str).str.strip()
        numeric_values = pd.to_numeric(raw_values.str.replace(",", "", regex=False), errors="coerce")
        invalid_count = int(((raw_values != "") & numeric_values.isna()).sum())
        negative_count = int((numeric_values < 0).sum())

        if invalid_count:
            board["Errors"].append(f"{column} has {invalid_count} non-numeric value(s).")
        if negative_count:
            board["Warnings"].append(f"{column} has {negative_count} negative value(s).")

    if "email_open_rate" in df.columns:
        email_values = pd.to_numeric(df["email_open_rate"].astype(str).str.strip(), errors="coerce")
        invalid_range = int(((email_values < 0) | (email_values > 100)).sum())
        if invalid_range:
            board["Warnings"].append("email_open_rate should stay between 0 and 100.")

    if "stage" in df.columns:
        stages = df["stage"].astype(str).str.strip()
        unknown_stages = sorted(stage for stage in stages.unique() if stage and stage not in STAGE_ORDER)
        for stage in unknown_stages:
            board["Warnings"].append(f"Unknown pipeline stage: {stage}.")

    if "last_activity_days" in df.columns:
        last_activity = pd.to_numeric(df["last_activity_days"].astype(str).str.strip(), errors="coerce")
        stale_count = int((last_activity >= 5).sum())
        if stale_count:
            board["To Do"].append(f"Follow up with {stale_count} lead(s) inactive for 5+ days.")

    if "email_open_rate" in df.columns:
        email_values = pd.to_numeric(df["email_open_rate"].astype(str).str.strip(), errors="coerce")
        low_email_count = int((email_values < 50).sum())
        if low_email_count:
            board["To Do"].append(f"Review outreach for {low_email_count} lead(s) with email open rate below 50%.")

    if not board["Errors"]:
        board["Errors"].append("No blocking CSV errors found.")
    if not board["Warnings"]:
        board["Warnings"].append("No data warnings found.")
    if not board["To Do"]:
        board["To Do"].append("No follow-up tasks generated from this dataset.")

    return board


def score_to_priority(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 60:
        return "Medium"
    return "Low"


def calculate_lead_score(row: pd.Series) -> float:
    budget_component = min(row["budget"] / 70000.0, 1.0) * 28
    value_component = min(row["deal_value"] / 85000.0, 1.0) * 19
    activity_component = max(0.0, (10.0 - min(row["last_activity_days"], 10.0)) / 10.0) * 17
    engagement_component = min(row["email_open_rate"] / 100.0, 1.0) * 12
    meetings_component = min(row["meetings_booked"] / 3.0, 1.0) * 12
    size_component = min(row["employees"] / 500.0, 1.0) * 8

    stage_bonus = {
        "New": 2,
        "Contacted": 5,
        "Qualified": 8,
        "Proposal": 12,
        "Negotiation": 16,
        "Won": 20,
        "Lost": 0,
    }.get(str(row["stage"]), 3)

    score = (
        budget_component
        + value_component
        + activity_component
        + engagement_component
        + meetings_component
        + size_component
        + stage_bonus
    )
    return round(min(score, 100.0), 1)


def calculate_conversion_probability(score: float, stage: str) -> float:
    stage_modifier = {
        "New": -12,
        "Contacted": -6,
        "Qualified": 0,
        "Proposal": 6,
        "Negotiation": 10,
        "Won": 15,
        "Lost": -25,
    }.get(stage, 0)
    probability = max(3.0, min(99.0, score + stage_modifier))
    return round(probability, 1)


def build_pricing_recommendation(row: pd.Series, conversion_probability: float) -> str:
    base_value = row["deal_value"]
    if conversion_probability >= 80:
        price = base_value * 1.08
        return f"Premium proposal: {format_currency(price)}"
    if conversion_probability >= 60:
        price = base_value
        return f"Standard proposal: {format_currency(price)}"
    price = max(row["budget"] * 0.95, base_value * 0.9)
    return f"Entry offer: {format_currency(price)}"


def build_strategy_note(row: pd.Series, conversion_probability: float) -> str:
    if conversion_probability >= 80:
        return "Fast-track this deal with a proposal call and decision timeline."
    if conversion_probability >= 60:
        return "Keep momentum with a tailored demo and pricing follow-up."
    return "Re-qualify needs, budget, and authority before deeper sales effort."


def prepare_leads(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=REQUIRED_COLUMNS
            + OPTIONAL_COLUMNS
            + ["score", "priority", "conversion_probability", "pricing_recommendation", "strategy_note"]
        )

    prepared = df.copy()

    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if column not in prepared.columns:
            prepared[column] = ""

    for column in NUMERIC_COLUMNS:
        prepared[column] = prepared[column].apply(normalize_number)

    for column in TEXT_COLUMNS + OPTIONAL_COLUMNS:
        prepared[column] = prepared[column].astype(str).str.strip()
        prepared[column] = prepared[column].replace({"": "Unknown"})

    prepared["bitrix_owner"] = prepared["bitrix_owner"].where(
        prepared["bitrix_owner"] != "Unknown",
        prepared["manager"],
    )
    prepared["score"] = prepared.apply(calculate_lead_score, axis=1)
    prepared["priority"] = prepared["score"].apply(score_to_priority)
    prepared["conversion_probability"] = prepared.apply(
        lambda row: calculate_conversion_probability(row["score"], str(row["stage"])),
        axis=1,
    )
    prepared["pricing_recommendation"] = prepared.apply(
        lambda row: build_pricing_recommendation(row, row["conversion_probability"]),
        axis=1,
    )
    prepared["strategy_note"] = prepared.apply(
        lambda row: build_strategy_note(row, row["conversion_probability"]),
        axis=1,
    )
    prepared["stage"] = pd.Categorical(prepared["stage"], categories=STAGE_ORDER, ordered=True)
    prepared = prepared.sort_values(["conversion_probability", "deal_value"], ascending=[False, False]).reset_index(drop=True)
    return prepared


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def priority_class(priority: str) -> str:
    return {
        "High": "priority-high",
        "Medium": "priority-medium",
        "Low": "priority-low",
    }.get(priority, "priority-low")


def filter_leads(
    leads: pd.DataFrame,
    selected_managers: list[str],
    selected_stages: list[str],
    selected_industries: list[str],
    conversion_range: tuple[int, int],
    budget_range: tuple[int, int],
) -> pd.DataFrame:
    if leads.empty:
        return leads

    filtered = leads.copy()
    if selected_managers:
        filtered = filtered[filtered["manager"].isin(selected_managers)]
    if selected_stages:
        filtered = filtered[filtered["stage"].astype(str).isin(selected_stages)]
    if selected_industries:
        filtered = filtered[filtered["industry"].isin(selected_industries)]

    filtered = filtered[
        (filtered["conversion_probability"] >= conversion_range[0])
        & (filtered["conversion_probability"] <= conversion_range[1])
        & (filtered["budget"] >= budget_range[0])
        & (filtered["budget"] <= budget_range[1])
    ]
    return filtered.reset_index(drop=True)


def render_sidebar() -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.markdown("## CRM Controls")
    st.sidebar.caption("Use the bundled sample CSV or upload your own Bitrix-style lead export.")

    use_sample = st.sidebar.toggle("Use sample dataset", value=False)
    uploaded_file = st.sidebar.file_uploader("Upload leads CSV", type=["csv"])

    if st.sidebar.button("Load data", use_container_width=True):
        started = time.perf_counter()
        if uploaded_file is not None:
            raw_leads = load_leads_from_upload(uploaded_file)
            leads = prepare_leads(raw_leads)
            st.session_state.dataset_label = f"Uploaded file: {uploaded_file.name}"
        elif use_sample:
            raw_leads = load_sample_leads()
            leads = prepare_leads(raw_leads)
            st.session_state.dataset_label = f"Sample file: {SAMPLE_LEADS_PATH.name}"
        else:
            raw_leads = pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
            leads = pd.DataFrame(
                columns=REQUIRED_COLUMNS
                + OPTIONAL_COLUMNS
                + ["score", "priority", "conversion_probability", "pricing_recommendation", "strategy_note"]
            )
            st.session_state.dataset_label = "No dataset loaded"

        st.session_state.loaded_leads = leads
        st.session_state.data_quality_board = build_data_quality_board(raw_leads)
        st.session_state.response_seconds = time.perf_counter() - started

    leads = st.session_state.loaded_leads

    managers = sorted(leads["manager"].astype(str).unique().tolist()) if not leads.empty else []
    stages = [stage for stage in STAGE_ORDER if not leads.empty and stage in leads["stage"].astype(str).tolist()]
    industries = sorted(leads["industry"].astype(str).unique().tolist()) if not leads.empty else []

    selected_managers = st.sidebar.multiselect("Manager", managers, default=managers)
    selected_stages = st.sidebar.multiselect("Pipeline stage", stages, default=stages)
    selected_industries = st.sidebar.multiselect("Industry", industries, default=industries)

    conversion_range = st.sidebar.slider("Conversion probability", 0, 100, (40, 100), 5)
    max_budget = int(math.ceil(leads["budget"].max() / 1000.0) * 1000) if not leads.empty else 100000
    budget_range = st.sidebar.slider("Budget range", 0, max_budget, (0, max_budget), 1000)

    filtered = filter_leads(
        leads,
        selected_managers,
        selected_stages,
        selected_industries,
        conversion_range,
        budget_range,
    )

    st.sidebar.divider()
    st.sidebar.markdown("### Data Status")
    st.sidebar.write("Dataset loaded" if not leads.empty else "No dataset loaded")
    st.sidebar.write(f"Total leads: {len(leads)}")
    st.sidebar.write(f"Visible leads: {len(filtered)}")
    st.sidebar.write(st.session_state.dataset_label)
    if st.session_state.response_seconds is not None:
        st.sidebar.success(f"Response time: {st.session_state.response_seconds:.4f} sec")

    return leads, filtered


def render_hero(total_leads: int) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <h1>Lead Scoring &amp; Pipeline Manager</h1>
            <p>
                Sales analytics dashboard for ranking opportunities, monitoring pipeline health,
                and reviewing CRM-ready leads with a Bitrix24-friendly structure.
                The current workspace includes <strong>{total_leads}</strong> leads ready for demo.
            </p>
            <div class="hero-tags">
                <span class="hero-tag">Project 3 Alignment</span>
                <span class="hero-tag">KPI Metrics</span>
                <span class="hero-tag">Sidebar Filters</span>
                <span class="hero-tag">Charts + Table</span>
                <span class="hero-tag">Bitrix24 Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(filtered: pd.DataFrame) -> None:
    total_leads = len(filtered)
    pipeline_value = filtered["deal_value"].sum() if total_leads else 0.0
    high_priority = int((filtered["priority"] == "High").sum()) if total_leads else 0
    conversion_rate = filtered["conversion_probability"].mean() if total_leads else 0.0
    avg_budget = filtered["budget"].mean() if total_leads else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High Priority Leads", high_priority)
    c2.metric("Conversion Rate", f"{conversion_rate:.1f}%")
    c3.metric("Total Value", format_currency(pipeline_value))
    c4.metric("Average Budget", format_currency(avg_budget))

    st.caption(f"Visible leads after filters: {total_leads}")


def render_overview_cards(filtered: pd.DataFrame) -> None:
    if filtered.empty:
        return

    top_company = filtered.iloc[0]["company"]
    top_probability = filtered.iloc[0]["conversion_probability"]
    fastest_manager = filtered.groupby("manager")["last_activity_days"].mean().sort_values().index[0]
    best_source = filtered.groupby("source")["conversion_probability"].mean().sort_values(ascending=False).index[0]

    left, middle, right = st.columns([1.1, 1, 1])

    with left:
        st.markdown(
            f"""
            <div class="section-card">
                <div class="section-title">Opportunity Radar</div>
                <div class="section-subtle">Fast read of the current pipeline after applying filters.</div>
                <div class="mini-grid">
                    <div class="mini-stat">
                        <div class="mini-label">Top lead</div>
                        <div class="mini-value">{top_company}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Top conversion</div>
                        <div class="mini-value">{top_probability:.1f}%</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Fastest manager</div>
                        <div class="mini-value">{fastest_manager}</div>
                    </div>
                    <div class="mini-stat">
                        <div class="mini-label">Best source</div>
                        <div class="mini-value">{best_source}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with middle:
        stage_counts = (
            filtered.groupby(filtered["stage"].astype(str), observed=False)
            .size()
            .reindex(STAGE_ORDER, fill_value=0)
        )
        st.markdown('<div class="section-card"><div class="section-title">Stage Distribution</div><div class="section-subtle">Where the visible deals currently sit in the funnel.</div>', unsafe_allow_html=True)
        st.bar_chart(stage_counts)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        manager_scores = filtered.groupby("manager")["conversion_probability"].mean().sort_values(ascending=False)
        st.markdown('<div class="section-card"><div class="section-title">Manager Performance</div><div class="section-subtle">Average conversion probability by owner.</div>', unsafe_allow_html=True)
        st.bar_chart(manager_scores)
        st.markdown("</div>", unsafe_allow_html=True)


def render_top_leads(filtered: pd.DataFrame) -> None:
    st.markdown("### Top Opportunities")
    if filtered.empty:
        st.info("No leads available for ranking.")
        return

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    for _, row in filtered.head(5).iterrows():
        fill = min(max(float(row["conversion_probability"]), 0.0), 100.0)
        st.markdown(
            f"""
            <div class="lead-strip">
                <div>
                    <strong>{row["company"]}</strong>
                    <div class="lead-meta">{row["lead_id"]} | {row["manager"]} | {row["stage"]} | {row["industry"]}</div>
                </div>
                <div style="min-width: 15rem; width: 40%;">
                    <div class="bar-track"><div class="bar-fill" style="width:{fill}%;"></div></div>
                    <div class="lead-meta" style="margin-top:0.35rem;">Conversion {row["conversion_probability"]:.1f}% | Value {format_currency(row["deal_value"])}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_manager_snapshot(filtered: pd.DataFrame) -> None:
    st.markdown("### Manager Snapshot")
    if filtered.empty:
        st.info("No manager data available.")
        return

    grouped = (
        filtered.groupby("manager")
        .agg(
            leads=("lead_id", "count"),
            conversion_probability=("conversion_probability", "mean"),
            pipeline_value=("deal_value", "sum"),
        )
        .sort_values(["conversion_probability", "pipeline_value"], ascending=[False, False])
        .reset_index()
    )
    grouped["conversion_probability"] = grouped["conversion_probability"].map(lambda value: f"{value:.1f}%")
    grouped["pipeline_value"] = grouped["pipeline_value"].map(format_currency)
    st.dataframe(grouped, use_container_width=True, hide_index=True)


def build_recommendations(filtered: pd.DataFrame) -> list[str]:
    if filtered.empty:
        return []

    recommendations: list[str] = []

    stale = filtered[filtered["last_activity_days"] >= 5]
    if not stale.empty:
        recommendations.append(
            f"{len(stale)} lead(s) show inactivity of 5+ days. Re-engagement tasks should be prioritized this week."
        )

    high_value = filtered[(filtered["priority"] == "High") & (filtered["stage"].astype(str).isin(["Proposal", "Negotiation"]))]
    if not high_value.empty:
        recommendations.append(
            f"{len(high_value)} high-priority deal(s) are already in proposal or negotiation. These are the closest revenue opportunities."
        )

    weak_engagement = filtered[filtered["email_open_rate"] < 50]
    if not weak_engagement.empty:
        recommendations.append(
            f"{len(weak_engagement)} lead(s) have low email engagement. Consider a call-first follow-up instead of another email."
        )

    top_source = filtered.groupby("source")["conversion_probability"].mean().sort_values(ascending=False).index[0]
    recommendations.append(
        f"Best-performing source in the current view is {top_source}. It would be a strong candidate for more budget allocation."
    )

    return recommendations[:4]


def render_insights(filtered: pd.DataFrame) -> None:
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Pipeline Insights")
        recommendations = build_recommendations(filtered)
        if not recommendations:
            st.info("Insights will appear once leads are loaded.")
        else:
            items = "".join(f"<li>{item}</li>" for item in recommendations)
            st.markdown(
                f"""
                <div class="section-card">
                    <div class="section-title">Recommended Actions</div>
                    <div class="section-subtle">Heuristic guidance generated from the visible pipeline state.</div>
                    <ul class="insight-list">{items}</ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown("### Priority Mix")
        if filtered.empty:
            st.info("No priority data available.")
        else:
            mix = filtered["priority"].value_counts().reindex(["High", "Medium", "Low"], fill_value=0)
            mix_df = pd.DataFrame({"priority": mix.index, "leads": mix.values})
            st.dataframe(mix_df, use_container_width=True, hide_index=True)


def render_data_quality_board() -> None:
    st.markdown("### Data QA Board")
    board = st.session_state.data_quality_board
    columns = ["Errors", "Warnings", "To Do"]
    rendered_columns = []

    for column in columns:
        cards = "".join(f'<div class="qa-card">{item}</div>' for item in board.get(column, []))
        rendered_columns.append(
            f"""
            <div class="qa-column">
                <div class="qa-title">{column}</div>
                {cards}
            </div>
            """
        )

    st.markdown(
        f"""
        <div class="section-card">
            <div class="section-subtle">
                Trello-style checklist for CSV problems and follow-up tasks generated during analysis.
            </div>
            <div class="qa-board">
                {''.join(rendered_columns)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_problem_reporter() -> None:
    st.markdown("### Client Problem Reporter")
    st.caption("Submitted client problems are sent to the configured Trello To Do list.")

    api_status = "Connected" if is_trello_configured() else "Not configured"
    c1, c2, c3 = st.columns(3)
    c1.metric("API Provider", "Trello")
    c2.metric("Target List", "To Do")
    c3.metric("API Status", api_status)

    with st.form("problem_report_form", clear_on_submit=True):
        title = st.text_input("Problem title", placeholder="Example: CSV upload fails for client file")
        description = st.text_area(
            "Problem description",
            placeholder="Describe what happened, expected result, and any useful client details.",
            height=120,
        )
        priority = st.selectbox("Priority", ["Medium", "High", "Low"])
        submitted = st.form_submit_button("Send to To Do list", use_container_width=True)

    if submitted:
        clean_title = title.strip()
        clean_description = description.strip()

        if not clean_title or not clean_description:
            st.error("Please enter both a problem title and description.")
            return

        success, result = create_trello_card(clean_title, clean_description, priority)
        ticket = {
            "title": clean_title,
            "priority": priority,
            "status": "Sent to Trello" if success else "Saved locally",
            "result": result,
        }
        st.session_state.support_tickets.insert(0, ticket)

        if success:
            st.success(f"Problem was added to Trello: {result}")
        else:
            st.warning(result)

    if st.session_state.support_tickets:
        st.markdown("#### Latest Submitted Problems")
        st.dataframe(pd.DataFrame(st.session_state.support_tickets), use_container_width=True, hide_index=True)


def render_lead_table(filtered: pd.DataFrame) -> None:
    st.markdown("### Lead Table")
    if filtered.empty:
        st.warning("No leads match the selected filters.")
        return

    display = filtered[
        [
            "lead_id",
            "company",
            "manager",
            "industry",
            "stage",
            "budget",
            "deal_value",
            "conversion_probability",
            "pricing_recommendation",
            "priority",
            "bitrix_owner",
        ]
    ].copy()
    display["budget"] = display["budget"].map(format_currency)
    display["deal_value"] = display["deal_value"].map(format_currency)
    display["conversion_probability"] = display["conversion_probability"].map(lambda value: f"{value:.1f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_selected_lead(filtered: pd.DataFrame) -> None:
    st.markdown("### Lead Detail View")
    if filtered.empty:
        st.info("Select filters that keep at least one lead visible.")
        return

    selected_id = st.selectbox("Choose lead", filtered["lead_id"].tolist())
    selected = filtered[filtered["lead_id"] == selected_id].iloc[0]

    priority_badge = priority_class(selected["priority"])
    win_prob = int(selected["conversion_probability"])

    st.markdown(
        f"""
        <div class="section-card">
            <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;flex-wrap:wrap;">
                <div>
                    <div class="section-title">{selected["company"]}</div>
                    <div class="section-subtle">{selected["lead_id"]} | {selected["industry"]} | {selected["region"]}</div>
                    <span class="priority-pill {priority_badge}">{selected["priority"]} Priority</span>
                </div>
                <div style="min-width:16rem;">
                    <div class="mini-label">Conversion probability</div>
                    <div class="bar-track" style="margin-top:0.35rem;"><div class="bar-fill" style="width:{win_prob}%;"></div></div>
                    <div class="lead-meta" style="margin-top:0.4rem;">{win_prob}% based on current lead signals</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stage", str(selected["stage"]))
    c2.metric("Conversion", f"{selected['conversion_probability']:.1f}%")
    c3.metric("Deal Value", format_currency(selected["deal_value"]))
    c4.metric("Last Activity", f"{int(selected['last_activity_days'])} day(s)")

    detail_df = pd.DataFrame(
        [
            ("Manager", selected["manager"]),
            ("Bitrix Owner", selected["bitrix_owner"]),
            ("Source", selected["source"]),
            ("Budget", format_currency(selected["budget"])),
            ("Pricing Recommendation", selected["pricing_recommendation"]),
            ("Sales Strategy", selected["strategy_note"]),
            ("Meetings Booked", int(selected["meetings_booked"])),
            ("Email Open Rate", f"{selected['email_open_rate']:.0f}%"),
            ("Employees", int(selected["employees"])),
        ],
        columns=["Field", "Value"],
    )
    with st.expander("Open lead details", expanded=True):
        st.dataframe(detail_df, use_container_width=True, hide_index=True)


def main() -> None:
    start_health_server()
    init_state()
    st.markdown(CSS, unsafe_allow_html=True)

    all_leads, filtered = render_sidebar()
    render_hero(len(all_leads))

    st.caption(
        "This application is fully repurposed for Project 3: Lead Scoring & Pipeline Manager with Bitrix24-friendly fields, KPI cards, filters, charts, and a sortable lead table."
    )

    if st.session_state.response_seconds is not None:
        st.info(f"Data loaded in {st.session_state.response_seconds:.4f} seconds.")

    render_problem_reporter()

    if filtered.empty:
        st.info("Choose a dataset in the sidebar and click `Load data`.")
        return

    render_metrics(filtered)
    render_overview_cards(filtered)
    left, right = st.columns([1.05, 0.95])
    with left:
        render_top_leads(filtered)
    with right:
        render_manager_snapshot(filtered)
    render_insights(filtered)
    render_data_quality_board()
    render_lead_table(filtered)
    render_selected_lead(filtered)


if __name__ == "__main__":
    main()
