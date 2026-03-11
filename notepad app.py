import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

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

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Big Data AI Analytics Platform",
    page_icon="🤖",
    layout="wide"
)

# ======================================================
# MOBILE SIDEBAR ANIMATION
# ======================================================
st.markdown("""
<style>
@media (max-width: 768px) {
  section[data-testid="stSidebar"] {
    transform: translateX(-100%);
    transition: transform 0.3s ease-in-out;
    position: fixed;
    z-index: 999;
  }
  section[data-testid="stSidebar"]:hover {
    transform: translateX(0);
  }
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# LOGO
# ======================================================
LOGO_URL = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"

# ======================================================
# SESSION STATE
# ======================================================
defaults = {
    "page": "Login",
    "logged_in": False,
    "role": None,
    "datasets": {},
    "active_dataset": None,
    "chat": [],
    "trained_model": None,
    "model_features": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ======================================================
# AUTO TASK DETECTION
# ======================================================
def detect_ml_task(y):
    return "classification" if y.nunique() <= 10 else "regression"

# ======================================================
# MODEL FACTORY
# ======================================================
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

# ======================================================
# TRAIN & EVALUATE
# ======================================================
def train_ml_model(df, target_col, model_name="rf"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    for col in X.select_dtypes(include="object").columns:
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

    return model, task, score

# ======================================================
# CHAT COMMAND PARSER
# ======================================================
def parse_chat_command(cmd, df):
    cmd = cmd.lower()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    chart, x, y, target = None, None, None, None
    predict = "predict" in cmd or "train" in cmd
    model_type = "rf"

    if "linear" in cmd: model_type = "linear"
    elif "random" in cmd: model_type = "rf"
    elif "xgb" in cmd: model_type = "xgb"
    elif "svm" in cmd: model_type = "svm"
    elif "knn" in cmd: model_type = "knn"
    elif "extra" in cmd: model_type = "extra"
    elif "gb" in cmd: model_type = "gb"

    if "bar" in cmd: chart = "Bar"
    elif "histogram" in cmd: chart = "Histogram"
    elif "scatter" in cmd: chart = "Scatter"

    for col in df.columns:
        if col.lower() in cmd:
            if not x:
                x = col
            else:
                y = col

    for col in num_cols:
        if col.lower() in cmd:
            target = col

    return chart, x, y, predict, target, model_type

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.image(LOGO_URL, width=120)
st.sidebar.header("🗂 Dataset Manager")

uploaded = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"],
    accept_multiple_files=True
) if st.session_state.role == "Admin" else []

for f in uploaded:
    st.session_state.datasets[f.name] = pd.read_csv(f)

if st.session_state.datasets:
    st.session_state.active_dataset = st.sidebar.selectbox(
        "Select Dataset",
        list(st.session_state.datasets.keys())
    )

df = (
    st.session_state.datasets.get(st.session_state.active_dataset)
    if st.session_state.active_dataset
    else None
)

# ======================================================
# LOGIN PAGE
# ======================================================
if st.session_state.page == "Login":
    st.image(LOGO_URL, width=100)
    st.title("Big Data AI Analytics")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "admin123":
            st.session_state.logged_in = True
            st.session_state.role = "Admin"
            st.session_state.page = "Dashboard"
            st.rerun()
        elif u == "user" and p == "user123":
            st.session_state.logged_in = True
            st.session_state.role = "User"
            st.session_state.page = "Dashboard"
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# ======================================================
# DATA PREVIEW
# ======================================================
if df is not None:
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

# ======================================================
# CHATBOT
# ======================================================
st.divider()
st.subheader("💬 AI Chatbot → Charts → AutoML")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_cmd = st.chat_input("Example: train random forest on total_bill")

if user_cmd and df is not None:
    st.session_state.chat.append({"role": "user", "content": user_cmd})

    chart, x, y, do_predict, target, model_type = parse_chat_command(user_cmd, df)

    if do_predict and target:
        model, task, score = train_ml_model(df, target, model_type)

        st.session_state.trained_model = model
        st.session_state.model_features = df.drop(columns=[target]).columns.tolist()

        response = f"""
✅ **Auto ML Completed**

🔹 Task Type: **{task.upper()}**
🔹 Model Used: **{model.__class__.__name__}**
🔹 Score: **{round(score, 3)}**
"""
        st.session_state.chat.append({
            "role": "assistant",
            "content": response
        })

    elif chart and x:
        fig, ax = plt.subplots()
        if chart == "Histogram":
            ax.hist(df[x], bins=30)
        elif chart == "Bar":
            df[x].value_counts().head(10).plot(kind="bar", ax=ax)
        elif chart == "Scatter" and y:
            ax.scatter(df[x], df[y])
        st.pyplot(fig)

    else:
        try:
            out = subprocess.run(
                ["ollama", "run", "tinyllama", user_cmd],
                stdout=subprocess.PIPE,
                encoding="utf-8"
            )
            st.session_state.chat.append({
                "role": "assistant",
                "content": out.stdout
            })
        except:
            st.session_state.chat.append({
                "role": "assistant",
                "content": "⚠️ Ollama not running"
            })

    st.rerun()































import streamlit as st

st.title("LLM Chatbot")
st.write("Streamlit is working!")



















