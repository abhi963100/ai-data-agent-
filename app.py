import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor
import difflib
import inspect


# ==============================
# STREAMLIT SESSION MEMORY
# ==============================
if "memory" not in st.session_state:
    st.session_state.memory = []

# ==============================
# FUNCTIONS
# ==============================
def handle_llm_chat(user_input, df):
    st.session_state.memory.append({
        "role": "user",
        "content": user_input
    })

    if df is None:
        return {
            "tool": "explain",
            "params": {"text": "Please upload a dataset first."}
        }

    decision = llm_reasoning(user_input, df, st.session_state.memory)
    if not decision or "tool" not in decision:
        fallback_tool = infer_intent(user_input)
        decision = {"tool": fallback_tool, "params": {}}

    if not decision or "tool" not in decision:
        return {
            "tool": "explain",
            "params": {"text": "Invalid LLM response"}
        }

    return decision

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI powered llm chatbot ",
    page_icon="🤖",
    layout="wide"
)
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"


# ======================================================
# ULTRA ADVANCED THEME HANDLER
# ======================================================
def apply_theme(theme: str):

    dark_theme = theme == "dark"

    bg_main = "#0f172a" if dark_theme else "#f8fafc"
    text_color = "#e2e8f0" if dark_theme else "#0f172a"
    card_bg = "rgba(255,255,255,0.06)" if dark_theme else "white"
    border_color = "rgba(255,255,255,0.08)" if dark_theme else "rgba(0,0,0,0.06)"
    gradient_1 = "#3b82f6"
    gradient_2 = "#8b5cf6"

    st.markdown(f"""
    <style>

    /* ===============================
       GLOBAL STYLING
    =============================== */
    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system;
        background-color: {bg_main};
        color: {text_color};
        transition: all 0.4s ease-in-out;
    }}

    .main {{
        background: radial-gradient(circle at 20% 10%, rgba(99,102,241,0.15), transparent 40%),
                    radial-gradient(circle at 80% 90%, rgba(139,92,246,0.15), transparent 40%);
        animation: fadeIn 1.2s ease-in-out;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* ===============================
       SIDEBAR
    =============================== */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {bg_main}, {bg_main});
        border-right: 1px solid {border_color};
        backdrop-filter: blur(14px);
    }}

    section[data-testid="stSidebar"] * {{
        color: {text_color};
    }}

    /* ===============================
       HEADERS
    =============================== */
    h1 {{
        font-weight: 800;
        background: linear-gradient(90deg, {gradient_1}, {gradient_2});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.03em;
    }}

    h2, h3 {{
        font-weight: 700;
    }}

    /* ===============================
       BUTTONS
    =============================== */
    .stButton > button {{
        background: linear-gradient(135deg, {gradient_1}, {gradient_2});
        color: white;
        border-radius: 12px;
        padding: 0.6em 1.3em;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(139,92,246,0.35);
    }}

    .stButton > button:active {{
        transform: scale(0.97);
    }}

    /* ===============================
       INPUTS
    =============================== */
    input, textarea, select {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border-radius: 12px !important;
        border: 1px solid {border_color} !important;
        transition: all 0.3s ease;
    }}

    input:focus, textarea:focus {{
        border: 1px solid {gradient_2} !important;
        box-shadow: 0 0 0 2px rgba(139,92,246,0.25);
    }}

    /* ===============================
       CARDS / METRICS
    =============================== */
    div[data-testid="metric-container"],
    .stDataFrame {{
        background: {card_bg};
        backdrop-filter: blur(16px);
        border-radius: 16px;
        padding: 15px;
        border: 1px solid {border_color};
        transition: transform 0.3s ease;
    }}

    div[data-testid="metric-container"]:hover {{
        transform: translateY(-4px);
    }}

    /* ===============================
       SCROLLBAR
    =============================== */
    ::-webkit-scrollbar {{
        width: 8px;
    }}

    ::-webkit-scrollbar-track {{
        background: transparent;
    }}

    ::-webkit-scrollbar-thumb {{
        background: linear-gradient({gradient_1}, {gradient_2});
        border-radius: 10px;
    }}

    /* ===============================
       TOAST / SUCCESS
    =============================== */
    .stAlert {{
        border-radius: 12px;
        backdrop-filter: blur(8px);
        border: 1px solid {border_color};
    }}

    /* ===============================
       REMOVE STREAMLIT BRANDING
    =============================== */
    header {{ visibility: hidden; height: 0px; }}
    footer {{ visibility: hidden; }}

    </style>
    """, unsafe_allow_html=True)
# ======================================================
# SESSION STATE DEFAULTS
# ======================================================
defaults = {
    "theme": "dark",
    "logged_in": False,
    "role": None,
    "datasets": {},
    "active_dataset": None,
    "chat": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

apply_theme(st.session_state.theme)

# ======================================================
# ANIMATED BACKGROUND LOGIN PAGE
# ======================================================
import streamlit as st

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:

    # ---------- CSS ----------
    st.markdown("""
    <style>
    header {
        height: 0px;
        visibility: hidden ;
    }

    .stApp {
        background-image: url("https://images.unsplash.com/photo-1518770660439-4636190af475");
        background-size: cover;
        background-position: center;
        animation: zoom 20s infinite alternate;
    }

    @keyframes zoom {
        0% { background-size: 100%; }
        100% { background-size: 115%; }
    }

    /* Dark overlay */
    .overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(2,6,23,0.85);
        z-index: -1;
    }

    /* Login Card */
    .login-box {
        background: rgba(15, 23, 42, 0.95);
        padding: 45px;
        border-radius: 20px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.7);
        width: 420px;
        animation: float 6s ease-in-out infinite;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .title {
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        color: #94a3b8;
        margin-bottom: 30px;
    }

    /* Login Image */
    .login-logo {
        display: flex;
        justify-content: center;
        margin-bottom: 20px;
    }

    .login-logo img {
        width: 90px;
        animation: pulse 3s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); opacity: 0.9; }
        50% { transform: scale(1.08); opacity: 1; }
        100% { transform: scale(1); opacity: 0.9; }
    }
    </style>

    <div class="overlay"></div>
    """, unsafe_allow_html=True)

    # ---------- Center Layout ----------
    left, center, right = st.columns([1, 1.2, 1])

    with center:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)

        # IMAGE IN PLACE OF BLANK SPACE
        st.markdown("""
        <div class="login-logo">
            <img src="https://cdn-icons-png.flaticon.com/512/4712/4712109.png">
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='title'>🤖 AI Analytics Login</div>", unsafe_allow_html=True)
        st.markdown("<div class='subtitle'>Big Data • Machine Learning • Insights</div>", unsafe_allow_html=True)

        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button(" Login", use_container_width=True):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.session_state.role = "Admin"
                st.rerun()
            elif username == "user" and password == "user123":
                st.session_state.logged_in = True
                st.session_state.role = "User"
                st.rerun()
            else:
                st.error(" Invalid username or password")

        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()
def render_output(result):
    """
    One shortcut renderer for LLM / tool output
    Graphs auto-display via Streamlit
    """
    if result is None:
        return

    if isinstance(result, dict):
        if "error" in result:
            st.error(result["error"])
        elif "text" in result:
            st.write(result["text"])
        else:
            st.write(result)
    else:
        st.write(result)

# ======================================================
# SIDEBAR
# ======================================================
# --- Logo
st.sidebar.image(LOGO_URL, width=90)
# --- Custom Sidebar Styling ---
st.sidebar.markdown("""
<style>

/* Sidebar container spacing */
section[data-testid="stSidebar"] {
    padding-top: 20px;
}

/* Sidebar Title */
.sidebar-title {
    font-size: 20px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 5px;
}

/* Divider */
.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.6), transparent);
    margin: 15px 0;
}

/* Status Badge */
.theme-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(99,102,241,0.15);
    color: #6366f1;
    margin-top: 5px;
}

/* Toggle Styling */
div[data-testid="stToggle"] > label {
    font-weight: 600;
}

/* Smooth hover */
section[data-testid="stSidebar"] * {
    transition: all 0.2s ease-in-out;
}

</style>
""", unsafe_allow_html=True)

# --- Sidebar Header ---
st.sidebar.markdown('<div class="sidebar-title">AI Analytics</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# --- Theme Toggle ---
theme_toggle = st.sidebar.toggle(
    " Dark / Light Mode",
    value=(st.session_state.theme == "light")
)

# --- Theme Logic ---
st.session_state.theme = "light" if theme_toggle else "dark"

# --- Theme Badge ---
current_theme = " Light Mode" if st.session_state.theme == "light" else " Dark Mode"
st.sidebar.markdown(f'<div class="theme-badge">{current_theme}</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

# --- Apply Theme ---
apply_theme(st.session_state.theme)

# ======================================================
# DATASET MANAGER
# ======================================================
st.sidebar.header("🗂 Dataset")

if st.session_state.role == "Admin":
    files = st.sidebar.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=True
    )
    for f in files:
        st.session_state.datasets[f.name] = pd.read_csv(f)

if st.session_state.datasets:
    st.session_state.active_dataset = st.sidebar.selectbox(
        "Select Dataset",
        list(st.session_state.datasets.keys())
    )
    df = st.session_state.datasets[st.session_state.active_dataset]
else:
    df = None

# ======================================================
# ML HELPERS
# ======================================================
def detect_ml_task(y):
    return "classification" if y.nunique() <= 10 else "regression"


def get_ml_model(model_name, task):
    if task == "regression":
        return {
            "linear": LinearRegression(),
            "rf": RandomForestRegressor(n_estimators=200),
            "xgb": XGBRegressor(n_estimators=200),
            "svm": SVR(),
            "knn": KNeighborsRegressor(),
            "gb": GradientBoostingRegressor(),
            "extra": ExtraTreesRegressor(n_estimators=300)
        }.get(model_name, LinearRegression())
    else:
        return {
            "logistic": LogisticRegression(max_iter=2000),
            "rf": RandomForestClassifier(n_estimators=200),
            "svm": SVC(),
            "knn": KNeighborsClassifier(),
            "gb": GradientBoostingClassifier(),
            "extra": ExtraTreesClassifier(n_estimators=300)
        }.get(model_name, LogisticRegression())


def train_ml_model(df, target, model_name="rf"):
    X = df.drop(columns=[target])
    y = df[target]

    for col in X.select_dtypes(include="object"):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    task = detect_ml_task(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = get_ml_model(model_name, task)
    model.fit(X_train, y_train)

    score = (
        r2_score(y_test, model.predict(X_test))
        if task == "regression"
        else accuracy_score(y_test, model.predict(X_test))
    )

    return task, score



# ======================================================
# DESCRIBE TOOL
# ======================================================
def describe_tool(df):
    st.dataframe(df.describe(), use_container_width=True)
    return "Dataset statistics displayed"


# ======================================================
# COUNT ROWS TOOL
# ======================================================
def count_rows_tool(df):
    return f"There are {df.shape[0]} rows in the dataset"


# ======================================================
# MULTI-STEP EXECUTOR
# ======================================================
def execute_plan(plan, df):

    if not plan or "steps" not in plan:
        return ["Could not generate execution plan"]

    results = []

    for step in plan["steps"]:
        tool = step.get("tool")
        params = step.get("params", {})

        if tool in TOOLS:
            decision = {"tool": tool, "params": params}
            result = execute_agent(decision, df)
            results.append(result)
    return results


# ======================================================
# CHART NORMALIZATION (ADD HERE)
# ======================================================
def normalize_chart(chart):
    if chart is None:
        return None

    chart = chart.lower()

    if chart in ["hist", "histogram", "distribution"]:
        return "histogram"

    if chart in ["bar", "bar chart", "barchart"]:
        return "bar"

    if chart in ["line", "line chart"]:
        return "line"

    if chart in ["scatter", "scatter plot"]:
        return "scatter"

    if chart in ["pie", "pie chart"]:
        return "pie"

    return chart

# ======================================================
# MAIN DASHBOARD
# ======================================================
st.title("🤖 Big Data AI Analytics")

if df is not None:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
else:
    st.info("⬅ Upload a CSV file to begin")

# ======================================================
#  REAL LLM ENGINE
# ======================================================

import subprocess
import json
import re

# ---------------------------
# LLM MEMORY
# ---------------------------
if "llm_memory" not in st.session_state:
    st.session_state.llm_memory = []

# ---------------------------
# STREAMING OLLAMA RESPONSE
# ---------------------------
def call_ollama_stream(prompt: str):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": True,
                "temperature": 0
            },
            stream=True,
            timeout=120
        )

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                if "response" in chunk:
                    yield chunk["response"]

    except Exception as e:
        yield f"\n Error: {e}"
# ---------------------------
# CALL OLLAMA (LLAMA 3)
# ---------------------------
import requests

def call_ollama(prompt: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False,
                "temperature": 0,
            },
            timeout=60
        )
        return response.json().get("response", "").strip()
    except Exception as e:
        return ""
# ============================
# LLM EXPLANATION LAYER
# ============================
def llm_explain(tool_result, user_input):
    """
    Uses LLM to explain tool results in natural language
    without changing factual values.
    """

    prompt = f"""
You are a helpful AI assistant.

User asked:
{user_input}

Here is the correct factual result (DO NOT CHANGE NUMBERS):
{tool_result}

Explain this clearly in simple, friendly language.
"""

    explanation = call_ollama(prompt)

    # Fallback safety
    if not explanation or not explanation.strip():
        return str(tool_result)

    return explanation.strip()
# ============================
# BUSINESS INSIGHT LAYER
# ============================
def business_insight(summary):

    prompt = f"""
You are a senior business analyst.

Convert the following analysis into executive-level insights.
Focus on risks, opportunities, and recommended actions.

Analysis:
{summary}
"""
    return call_ollama(prompt)

def recommend_next_step(analysis):

    prompt = f"""
You are an autonomous senior data scientist.

Based on the analysis below, recommend ONE clear next action.

Choose ONLY one:
1. More analysis
2. Feature engineering
3. Model improvement
4. Data cleaning
5. Deployment readiness

Respond in this format:

Next Step: <choice>
Reason: <short explanation>

Analysis:
{analysis}
"""

    response = call_ollama(prompt)

    if not response or not response.strip():
        return "Next Step: Further analysis\nReason: Default fallback due to empty response."

    return response.strip()

# ============================
# AUTO VERIFICATION LAYER
# ============================
import re

def auto_verify_answer(user_input, tool_name, tool_result, df):
    """
    Verifies critical answers against actual dataframe values.
    """

    text = user_input.lower()

    # ---- ROW COUNT ----
    if "row" in text or "how many" in text:
        real_count = len(df)
        return f"There are {real_count} rows in the dataset."

    # ---- COLUMN COUNT ----
    if "column" in text:
        real_cols = len(df.columns)
        return f"There are {real_cols} columns in the dataset."

    # ---- MISSING VALUES ----
    if "missing" in text:
        missing = df.isnull().sum().sum()
        return f"Total missing values in dataset: {missing}"

    final_answer = llm_explain(tool_result, user_input)
    return final_answer

# ============================
# SMART INTENT ROUTER
# ============================
def infer_intent(user_input: str):
    text = user_input.lower()

    if any(w in text for w in ["plot", "graph", "chart", "visual"]):
        return "plot"

    if any(w in text for w in ["train", "predict", "model"]):
        return "train"

    if any(w in text for w in ["describe", "summary", "statistics"]):
        return "describe"

    if any(w in text for w in ["rows", "count", "how many"]):
        return "count_rows"

    return "explain"

# ---------------------------
# LLM REASONING (STRICT JSON)
# ---------------------------
def llm_reasoning(user_input: str, df, memory):

    numeric_cols = list(df.select_dtypes(include="number").columns)
    categorical_cols = list(df.select_dtypes(exclude="number").columns)

    prompt = f"""
You are an AI DATA AGENT.

Return ONLY valid JSON in this exact format:

{{
  "tool": "plot|train|describe|count_rows|analyze|explain",

  "params": {{
    "column": null,
    "chart": null,
    "target": null,
    "model": null,
    "text": null
  }}
}}

Dataset info:
Numeric columns: {numeric_cols}
Categorical columns: {categorical_cols}

User request:
{user_input}

Rules:
- Plot numeric → histogram
- Plot categorical → bar
- Two numeric → scatter
- Train → tool=train
- Summary → tool=describe
- Count rows → tool=count_rows
- Otherwise → tool=explain

Return ONLY JSON.
"""

    # ===============================
    # CALL LLM
    # ===============================
    raw = call_ollama(prompt)

    # ===============================
    # FORCE JSON EXTRACTION
    # ===============================
    start = raw.find("{")
    end = raw.rfind("}")

    if start != -1 and end != -1:
        raw = raw[start:end + 1]

    # OPTIONAL: debug view
    st.sidebar.text_area("RAW LLM OUTPUT (SANITIZED)", raw, height=200)

    # ===============================
    # FINAL SAFE JSON PARSE
    # ===============================
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "tool": "explain",
            "params": {"text": "Invalid JSON from LLM"}
        }

# ======================================================
# MULTI-STEP LLM PLANNER
# ======================================================
def llm_plan(user_input, df):

    prompt = f"""
You are a planning agent.

Return ONLY valid JSON.

Format:
{{
  "steps": [
    {{ "tool": "describe", "params": {{}} }},
    {{ "tool": "plot", "params": {{ "column": "tenure" }} }},
    {{ "tool": "train", "params": {{ "target": "Churn" }} }}
  ]
}}

User request:
{user_input}

Dataset columns:
{list(df.columns)}

Rules:
- Summary → describe
- Plot → plot
- Train → train
- Multiple actions → multiple steps
- Return ONLY JSON
"""

    raw = call_ollama(prompt)

    import re, json
    match = re.search(r"\{.*\}", raw, re.S)

    if match:
        try:
            return json.loads(match.group())
        except:
            pass

    return None
# ============================
# SMART COLUMN MATCH
# ============================
def smart_column_match(col, df):
    if col in df.columns:
        return col

    match = difflib.get_close_matches(col, df.columns, n=1)
    return match[0] if match else None

# ---------------------------
# ADVANCED AGENT PLOT TOOL
# ---------------------------
def plot_tool(df, column=None, chart=None, explain=True):

    if chart is None:
        chart = "auto"

    if not column:
        return " Please specify a column to plot."

    column = smart_column_match(column, df)

    if not column:
        return "Column not found in dataset."

    explanation = []
    fig, ax = plt.subplots(figsize=(7, 4))

    # ==========================
    # SCATTER SUPPORT (x,y)
    # ==========================
    if "," in column:
        col1, col2 = [c.strip() for c in column.split(",")]

        if col1 in df.columns and col2 in df.columns:
            ax.scatter(df[col1], df[col2], alpha=0.7)
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            ax.set_title(f"{col1} vs {col2}")

            st.pyplot(fig)

            explanation.append(
                f"This scatter plot shows the relationship between **{col1}** and **{col2}**. "
                f"Each point represents one row in the dataset."
            )

            return "\n".join(explanation)

        else:
            return " Invalid columns for scatter plot."

    series = df[column].dropna()

    # ==========================
    # AUTO CHART DETECTION
    # ==========================
    if chart == "auto":
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() > 20:
                chart = "histogram"
            else:
                chart = "box"
        elif pd.api.types.is_datetime64_any_dtype(series):
            chart = "line"
        else:
            chart = "bar"

    # ==========================
    # HISTOGRAM
    # ==========================
    if chart == "histogram":
        series.hist(ax=ax, bins=25, color="#6366f1", edgecolor="white")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

        explanation.append(
            f"A histogram is used because **{column}** is a numeric column. "
            f"It shows the distribution of values and how frequently they occur."
        )

    # ==========================
    # BAR CHART
    # ==========================
    elif chart == "bar":
        series.value_counts().plot(kind="bar", ax=ax, color="#6366f1")
        ax.set_ylabel("Count")

        explanation.append(
            f"A bar chart is used because **{column}** is categorical. "
            f"It compares the frequency of each category."
        )

    # ==========================
    # BOX PLOT
    # ==========================
    elif chart == "box":
        ax.boxplot(series, vert=False)
        ax.set_xlabel(column)

        explanation.append(
            f"A box plot summarizes the distribution of **{column}**, "
            f"highlighting median, quartiles, and potential outliers."
        )

    # ==========================
    # LINE CHART (TIME SERIES)
    # ==========================
    elif chart == "line":
        ax.plot(series.values)
        ax.set_ylabel(column)
        ax.set_xlabel("Index")

        explanation.append(
            f"A line chart is used because **{column}** appears to be time-based or sequential. "
            f"It shows trends over time."
        )

    else:
        return f" Unsupported chart type: {chart}"

    ax.set_title(f"{chart.capitalize()} of {column}")
    st.pyplot(fig)

    # ==========================
    # STATISTICAL INSIGHT
    # ==========================
    if pd.api.types.is_numeric_dtype(series):
        explanation.append(
            f"Basic statistics — Mean: **{series.mean():.2f}**, "
            f"Min: **{series.min()}**, Max: **{series.max()}**."
        )

    # ==========================
    # FINAL RESPONSE
    # ==========================
    if explain:
        return " **Plot Explanation:**\n\n" + " ".join(explanation)
    else:
        return f"{chart.capitalize()} plotted for {column}"


# ============================
# SMART MODEL AUTO-SELECTION
# ============================
def auto_select_model(y):
    """
    Choose model automatically based on task
    """
    if y.nunique() <= 10:
        return "RandomForest"   # classification
    return "RandomForest"       # regression (safe default)

def train_tool(df, target=None, model=None):

    # ---------- TARGET AUTO-CORRECTION ----------
    if not target:
        return "Please specify a target column"

    target = smart_column_match(target, df)

    if not target:
        return "Target column not found in dataset"

    X = df.drop(columns=[target])
    y = df[target]

    # ---------- ENCODE CATEGORICAL ----------
    for c in X.select_dtypes(include="object"):
        X[c] = LabelEncoder().fit_transform(X[c].astype(str))

    # ---------- TASK DETECTION ----------
    task = "classification" if y.nunique() <= 10 else "regression"

    # ---------- MODEL AUTO-SELECTION ----------
    if not model:
        model = auto_select_model(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------- CLASSIFICATION ----------
    if task == "classification":
        mdl = {
            "RandomForest": RandomForestClassifier(),
            "LogisticRegression": LogisticRegression(max_iter=2000),
            "SVM": SVC()
        }.get(model, RandomForestClassifier())

        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        score = accuracy_score(y_test, preds)

        result = f"Model trained ({model}) | Accuracy: {round(score, 3)}"

    # ---------- REGRESSION ----------
    else:
        mdl = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor()
        }.get(model, RandomForestRegressor())

        mdl.fit(X_train, y_train)
        preds = mdl.predict(X_test)
        score = r2_score(y_test, preds)

        result = f"Model trained ({model}) | R2: {round(score, 3)}"

    # ---------- FEATURE IMPORTANCE (ADD HERE) ----------
    if hasattr(mdl, "feature_importances_"):
        importances = mdl.feature_importances_
        features = X.columns

        top = sorted(
            zip(features, importances),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        insight = "\nTop Features:\n"
        insight += "\n".join(
            f"- {f}: {round(i, 3)}" for f, i in top
        )

        result += insight

    return result

# ===============================
# ADVANCED ANALYTICAL ENGINE
# ===============================
from scipy.stats import ttest_ind
import numpy as np

def analytical_engine(df, target=None):

    insights = []
    numeric = df.select_dtypes(include="number")

    # ----------------------------------
    # 1️⃣ MULTIVARIATE CORRELATION SCAN
    # ----------------------------------
    if numeric.shape[1] > 1:
        corr = numeric.corr()
        strong = [
            (i, j, corr.loc[i, j])
            for i in corr.columns
            for j in corr.columns
            if i != j and abs(corr.loc[i, j]) > 0.7
        ]
        if strong:
            insights.append("🔗 Strong correlations detected:")
            for s in strong[:5]:
                insights.append(f"{s[0]} ↔ {s[1]} (corr={round(s[2],2)})")

    # ----------------------------------
    # 2️⃣ TREND DETECTION
    # ----------------------------------
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) > 10:
            trend = np.polyfit(range(len(series)), series, 1)[0]
            if abs(trend) > 0.01:
                insights.append(
                    f"📈 Trend detected in {col}: {'increasing' if trend > 0 else 'decreasing'}"
                )

    # ----------------------------------
    # 3️⃣ ANOMALY DETECTION (IQR)
    # ----------------------------------
    for col in numeric.columns:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            insights.append(f"🚨 {col} has {len(outliers)} anomalies")

    # ----------------------------------
    # 4️⃣ DATA LEAKAGE CHECK
    # ----------------------------------
    if target and target in numeric.columns:
        for col in numeric.columns:
            if col != target:
                corr_val = abs(df[col].corr(df[target]))
                if corr_val > 0.95:
                    insights.append(
                        f"⚠️ Possible data leakage: {col} highly correlated with target ({corr_val:.2f})"
                    )

    # ----------------------------------
    # 5️⃣ GROUP COMPARISON + HYPOTHESIS TEST
    # ----------------------------------
    cat_cols = df.select_dtypes(include="object").columns
    if target and target in numeric.columns and len(cat_cols) > 0:
        cat = cat_cols[0]
        groups = df[cat].unique()
        if len(groups) == 2:
            g1 = df[df[cat] == groups[0]][target]
            g2 = df[df[cat] == groups[1]][target]
            stat, p = ttest_ind(g1, g2, nan_policy="omit")
            insights.append(
                f"🧪 Hypothesis Test ({cat} vs {target}): p-value = {round(p,4)}"
            )
            if p < 0.05:
                insights.append("➡️ Statistically significant difference detected")

    # ----------------------------------
    # 6️⃣ FEATURE ENGINEERING SUGGESTIONS
    # ----------------------------------
    insights.append("🛠 Feature Engineering Suggestions:")
    insights.append("- Normalize skewed numeric features")
    insights.append("- Try interaction terms between correlated variables")
    insights.append("- Encode categorical variables using target encoding")

    return "\n".join(insights)

# ======================================================
# TOOL REGISTRY (DEFINE ONCE)
# ======================================================
TOOLS = {
    "plot": plot_tool,
    "train": train_tool,
    "describe": describe_tool,
    "count_rows": count_rows_tool,
    "explain": lambda df, text=None: text,
    "analyze": lambda df, target=None: analytical_engine(df, target)
}
#######################################
#######execute_agent
#######################################
def execute_agent(decision, df):
    tool_name = decision.get("tool")
    params = decision.get("params", {})

    # ------------------------------
    # AUTO TOOL CORRECTION
    # ------------------------------
    if tool_name not in TOOLS:
        possible = list(TOOLS.keys())
        match = difflib.get_close_matches(tool_name, possible, n=1, cutoff=0.6)
        if match:
            tool_name = match[0]
        else:
            return f"Tool '{tool_name}' not found. Available tools: {list(TOOLS.keys())}"

    tool_fn = TOOLS[tool_name]

    # ------------------------------
    # SAFE PARAM FILTERING
    # ------------------------------
    sig = inspect.signature(tool_fn)
    safe_params = {k: v for k, v in params.items() if k in sig.parameters}

    result = tool_fn(df, **safe_params)

    # ==================================================
    # 🔥 ANALYZE PIPELINE (ADD HERE)
    # ==================================================
    if tool_name == "analyze":

        target = safe_params.get("target")

        analysis = analytical_engine(df, target)
        business = business_insight(analysis)
        next_step = recommend_next_step(analysis)

        st.markdown("### 📊 Deep Analysis")
        st.write(analysis)

        st.markdown("### 💼 Business Insights")
        st.write(business)

        st.markdown("### 🚀 Recommended Next Step")
        st.write(next_step)

        return analysis  # return something for chat history


    # ------------------------------
    # DEFAULT FLOW (OTHER TOOLS)
    # ------------------------------
    return result



import inspect
import difflib
# ======================================================
# MULTI-STEP PLAN EXECUTOR
# ======================================================
def execute_plan(plan, df):

    if not plan or "steps" not in plan:
        return ["Could not generate execution plan"]

    results = []

    for step in plan["steps"]:
        tool = step.get("tool")
        params = step.get("params", {})

        decision = {"tool": tool, "params": params}
        result = execute_agent(decision, df)
        results.append(result)
    return results


# ---------------------------
# MAIN CHAT HANDLER
# ---------------------------
def handle_llm_chat(user_input, df):

    # Make sure memory exists
    if "memory" not in st.session_state:
        st.session_state.memory = []

    # Add user input to memory
    st.session_state.memory.append({
        "role": "user",
        "content": user_input
    })

    if df is None:
        return {
            "tool": "explain",
            "params": {"text": "Please upload a dataset first."}
        }

    # Pass memory to LLM reasoning
    decision = llm_reasoning(
        user_input,
        df,
        st.session_state.memory
    )

    print("Agent decision:", decision)

    if not decision or "tool" not in decision:
        return {
            "tool": "explain",
            "params": {"text": "Invalid LLM response"}
        }

    return decision
# ---------------------------
# STREAMING CHAT RESPONSE
# ---------------------------
def stream_chat_response(prompt):
    placeholder = st.empty()
    full_text = ""

    for token in call_ollama_stream(prompt):
        full_text += token
        placeholder.markdown(full_text)

    return full_text



# ------------------------------------------------------
#  CHAT UI
# ------------------------------------------------------
st.divider()
st.subheader("💬 AI LLM Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about your dataset (e.g. plot Age)")

if user_input:
    st.session_state.chat.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    plan = llm_plan(user_input, df)

    if plan:
        replies = execute_plan(plan, df)
        for r in replies:
            render_output(r)
            st.session_state.chat.append({
                "role": "assistant",
                "content": str(r)
            })
    else:
        decision = handle_llm_chat(user_input, df)

        # If explain → stream response
        if decision["tool"] == "explain":
            context_info = f"\nDataset has {df.shape[0]} rows and {df.shape[1]} columns."
            reply = stream_chat_response(user_input + context_info)

        else:
            reply = execute_agent(decision, df)

        render_output(reply)

        st.session_state.chat.append({
            "role": "assistant",
            "content": str(reply)
        })

# ======================================================
# SHORT-TERM CONVERSATION MEMORY
# ======================================================
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []


def add_to_memory(role, content):
    st.session_state.chat_memory.append(
        {"role": role, "content": content}
    )
    # keep last 6 messages only
    st.session_state.chat_memory = st.session_state.chat_memory[-6:]


# ======================================================
#  MULTI-STEP AGENT PLANNER
# ======================================================
def agent_plan(user_query, df):
    plan_prompt = f"""
You are a data analysis agent.

Create a step-by-step plan in JSON.
Allowed steps: health_check, plot, train, explain.

Return ONLY JSON.

User request: {user_query}
Columns: {list(df.columns)}
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=plan_prompt,
        capture_output=True,
        text=True
    )

# ======================================================
#  DATASET HEALTH
# ======================================================
if st.sidebar.button("🩺 Dataset Health Check"):
    if df is not None:
        health = {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "dtypes": df.dtypes.astype(str).to_dict()
        }

        st.subheader(" Dataset Health Report")
        st.json(health)

        explanation_prompt = f"""
Explain this dataset health in simple words:
{health}
"""

        explanation = subprocess.run(
            ["ollama", "run", "llama3"],
            input=explanation_prompt,
            capture_output=True,
            text=True
        )

        st.markdown("###  LLM Explanation")
        st.write(explanation.stdout)


# ======================================================
#  LLM-BASED MODEL RECOMMENDATION
# ======================================================
def recommend_model(df,target_col):
    prompt = f"""
You are an ML expert.

Based on dataset info:
Columns: {list(df.columns)}
Target: {target_col}

Return ONLY one model name from:
LinearRegression, LogisticRegression, RandomForest, GradientBoosting
"""

    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        text=True
    )

    return result.stdout.strip()

# ==============================
# TOOL AUTO-CORRECTION MAP
# ==============================
TOOL_ALIASES = {
    "plot": "plot_chart",
    "chart": "plot_chart",
    "graph": "plot_chart",
    "visualize": "plot_chart",

    "train": "train_model",
    "model": "train_model",

    "statistics": "stats",
    "summary": "stats",

    "explanation": "explain",
    "describe": "explain"
}
# ==============================
# ONE-SHOT SAFE PLOT TOOL
# ==============================
def plot_chart(df, column=None, chart=None, **kwargs):
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    # Auto-fix missing column
    if column is None:
        st.error("No column provided for plotting")
        return

    if column not in df.columns:
        st.error(f"Column '{column}' not found in dataset")
        return

    fig, ax = plt.subplots()
    series = df[column].dropna()

    # Explicit chart type from LLM
    if chart:
        chart = chart.lower()

    # Drop NA safely
    series = series.dropna()

    insight = []

    # ------------------------------------
    # SMART AUTO-DECISION ENGINE
    # ------------------------------------
    if chart is None or chart == "auto":

        if pd.api.types.is_numeric_dtype(series):

            # If many unique values → histogram
            if series.nunique() > 15:
                chart = "hist"
            else:
                chart = "box"

        elif pd.api.types.is_datetime64_any_dtype(series):
            chart = "line"

        else:
            chart = "bar"

    # ------------------------------------
    # BAR CHART
    # ------------------------------------
    if chart == "bar":
        counts = series.value_counts()
        counts.plot(kind="bar", ax=ax)

        insight.append(
            f"Bar chart used because '{column}' is categorical."
        )
        insight.append(
            f"Most frequent value: {counts.idxmax()} ({counts.max()} times)."
        )


    # ------------------------------------
    # LINE CHART
    # ------------------------------------
    elif chart == "line":
        if pd.api.types.is_numeric_dtype(series):
            series.plot(ax=ax)
            insight.append(
                f"Line chart shows trend in numeric column '{column}'."
            )
        else:
            series.value_counts().sort_index().plot(ax=ax)
            insight.append(
                f"Line chart used to show ordered categorical trend."
            )


    # ------------------------------------
    # HISTOGRAM
    # ------------------------------------
    elif chart == "hist":
        series.hist(ax=ax, bins=25)

        insight.append(
            f"Histogram shows distribution of numeric column '{column}'."
        )
        insight.append(
            f"Mean: {series.mean():.2f}, Median: {series.median():.2f}."
        )

        # Skew detection
        skew = series.skew()
        if skew > 0.5:
            insight.append("Distribution appears right-skewed.")
        elif skew < -0.5:
            insight.append("Distribution appears left-skewed.")
        else:
            insight.append("Distribution appears fairly symmetric.")


    # ------------------------------------
    # BOX PLOT
    # ------------------------------------
    elif chart == "box":
        ax.boxplot(series, vert=False)

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        insight.append(
            f"Box plot shows spread of '{column}'. Q1: {q1:.2f}, Q3: {q3:.2f}."
        )


    # ------------------------------------
    # SCATTER SUPPORT (x,y)
    # ------------------------------------
    elif chart == "scatter" and "," in column:
        col1, col2 = [c.strip() for c in column.split(",")]

        if col1 in df.columns and col2 in df.columns:
            ax.scatter(df[col1], df[col2], alpha=0.6)

            corr = df[[col1, col2]].corr().iloc[0, 1]

            insight.append(
                f"Scatter plot shows relationship between {col1} and {col2}."
            )
            insight.append(
                f"Correlation coefficient: {corr:.2f}."
            )


    else:
        return f"Unsupported chart type: {chart}"

    ax.set_title(f"{chart.capitalize()} chart of {column}")
    st.pyplot(fig)

    return "Insights:\n\n" + "\n".join(insight)
