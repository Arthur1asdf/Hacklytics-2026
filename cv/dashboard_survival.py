import pandas as pd
from pathlib import Path

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

if st is not None:
    cache_data = st.cache_data
else:
    def cache_data(func):
        return func

ALL_LIMBS = ["head", "chest", "torso", "left_arm", "right_arm", "left_leg", "right_leg"]
STATE_ORDER = ["green", "yellow", "red", "missing", "occluded", "unknown"]
LOG_DIR = Path(__file__).resolve().parent / "logs"


@cache_data
def load_run(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ["elapsed_s", "critical_load", "red_count", "yellow_count", "green_count", "missing_count", "occluded_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


def transition_matrix(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for limb in ALL_LIMBS:
        col = f"{limb}_state"
        if col not in df.columns:
            continue
        s = df[col].astype(str)
        t = pd.DataFrame({"from_state": s.shift(1), "to_state": s}).dropna()
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["from_state", "to_state", "count"])
    transitions = pd.concat(parts, ignore_index=True)
    out = transitions.value_counts().reset_index(name="count")
    return out


def main():
    if st is None:
        raise RuntimeError("Streamlit is not installed. Run: pip install -r requirements.txt")
    import plotly.express as px

    st.set_page_config(page_title="Survival Drivers Dashboard", layout="wide")
    st.title("Survival Drivers Dashboard")
    st.caption("Non-Gemini analytics from frame-level limb logs")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(LOG_DIR.glob("limb_run_*.csv"), reverse=True)
    if not files:
        st.warning("No log files found. Run `cvopen.py` with `ENABLE_DATA_LOGGING = True` first.")
        return

    selected = st.selectbox("Choose run", files, format_func=lambda p: p.name, index=0)
    df = load_run(selected)
    if df.empty:
        st.warning("Selected log is empty.")
        return

    st.subheader("Run KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Frames", int(len(df)))
    c2.metric("Avg Critical Load", f"{df['critical_load'].mean():.2f}")
    c3.metric("Red Frame Rate", f"{(df['red_count'].gt(0).mean()*100):.1f}%")
    c4.metric("Occluded Rate", f"{(df['occluded_count'].gt(0).mean()*100):.1f}%")

    st.subheader("Critical Pressure Timeline")
    timeline = df[["timestamp", "critical_load", "red_count", "yellow_count", "missing_count", "occluded_count"]].copy()
    fig_timeline = px.line(
        timeline,
        x="timestamp",
        y=["critical_load", "red_count", "yellow_count", "missing_count", "occluded_count"],
        markers=False,
    )
    fig_timeline.update_layout(legend_title_text="Metric")
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.subheader("Limb State Timeline")
    limb = st.selectbox("Limb", ALL_LIMBS, index=1)
    state_col = f"{limb}_state"
    conf_col = f"{limb}_conf"
    limb_df = df[["timestamp", state_col, conf_col]].copy()
    limb_df[state_col] = pd.Categorical(limb_df[state_col], categories=STATE_ORDER, ordered=True)
    fig_limb = px.scatter(
        limb_df,
        x="timestamp",
        y=state_col,
        color=state_col,
        size=conf_col,
        category_orders={state_col: STATE_ORDER},
        color_discrete_map={
            "green": "#2ca02c",
            "yellow": "#f1c40f",
            "red": "#e74c3c",
            "missing": "#7f8c8d",
            "occluded": "#34495e",
            "unknown": "#bdc3c7",
        },
    )
    st.plotly_chart(fig_limb, use_container_width=True)

    st.subheader("State Transition Heatmap")
    trans = transition_matrix(df)
    if trans.empty:
        st.info("Not enough transitions yet.")
    else:
        pivot = trans.pivot(index="from_state", columns="to_state", values="count").fillna(0)
        pivot = pivot.reindex(index=STATE_ORDER, columns=STATE_ORDER, fill_value=0)
        fig_heat = px.imshow(
            pivot,
            labels=dict(x="To", y="From", color="Count"),
            text_auto=True,
            aspect="auto",
            color_continuous_scale="YlOrRd",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Top Risk Windows")
    df_sorted = df.sort_values("critical_load", ascending=False).head(10)
    st.dataframe(
        df_sorted[
            ["timestamp", "critical_load", "red_count", "yellow_count", "missing_count", "occluded_count"]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Export")
    st.download_button(
        label="Download current run CSV",
        data=selected.read_bytes(),
        file_name=selected.name,
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
