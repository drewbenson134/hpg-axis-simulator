import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Avian HPG Axis and Photostimulation Simulator",
    page_icon="🐔",
    layout="wide",
)


SIM_DAYS = 280
BASELINE_PHOTOSTIM_AGE = 147
MATURE_BW_G = 1650


def logistic(x: np.ndarray | float, midpoint: float, steepness: float) -> np.ndarray | float:
    """Return a bounded sigmoid used for gradual biological responses."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def build_growth_curve(days: np.ndarray, hatch_weight_g: float, mature_weight_g: float) -> np.ndarray:
    """Approximate pullet growth using a logistic curve."""
    growth_fraction = logistic(days, midpoint=98, steepness=0.045)
    return hatch_weight_g + (mature_weight_g - hatch_weight_g) * growth_fraction


def calculate_body_weight_modifier(target_weight_pct: float) -> float:
    """Translate target body weight achievement into a reproductive readiness modifier."""
    deviation = (target_weight_pct - 100.0) / 100.0
    return float(np.clip(1.0 + 0.8 * deviation, 0.75, 1.15))


def calculate_light_stimulation(
    photoperiod_hours: float,
    light_intensity_lux: float,
    spectrum_factor: float,
) -> float:
    """Estimate the strength of photostimulatory input to the hypothalamus."""
    photoperiod_signal = logistic(photoperiod_hours, midpoint=12.8, steepness=1.25)
    intensity_signal = logistic(light_intensity_lux, midpoint=12.0, steepness=0.22)
    stimulation = photoperiod_signal * intensity_signal * spectrum_factor
    return float(np.clip(stimulation, 0.0, 1.2))


def calculate_hpg_activation(
    days: np.ndarray,
    body_weight_g: np.ndarray,
    target_weight_pct: float,
    photostim_age_days: int,
    photoperiod_before: float,
    photoperiod_after: float,
    intensity_lux: float,
    spectrum_factor: float,
    stress_index: float,
) -> pd.DataFrame:
    """Simulate HPG activation and downstream reproductive hormones."""
    photoperiod = np.where(days < photostim_age_days, photoperiod_before, photoperiod_after)
    light_drive = np.array(
        [
            calculate_light_stimulation(hours, intensity_lux, spectrum_factor)
            for hours in photoperiod
        ]
    )

    readiness = np.clip(body_weight_g / (0.93 * MATURE_BW_G), 0.3, 1.2)
    target_modifier = calculate_body_weight_modifier(target_weight_pct)
    weight_drive = np.clip(readiness * target_modifier, 0.25, 1.2)

    age_drive = logistic(days, midpoint=132, steepness=0.055)
    stress_modifier = np.clip(1.0 - 0.45 * (stress_index / 100.0), 0.55, 1.0)

    hpg_signal = np.clip(light_drive * weight_drive * age_drive * stress_modifier, 0.0, 1.0)
    gnrh = 10 + 90 * logistic(hpg_signal, midpoint=0.42, steepness=9.0)
    lh = 5 + 95 * logistic(hpg_signal, midpoint=0.48, steepness=9.5)
    fsh = 20 + 70 * logistic(hpg_signal, midpoint=0.36, steepness=8.5)
    estradiol = 15 + 185 * logistic(hpg_signal, midpoint=0.55, steepness=10.0)
    ovarian_state = 100 * logistic(hpg_signal, midpoint=0.58, steepness=10.5)

    return pd.DataFrame(
        {
            "day": days,
            "body_weight_g": body_weight_g,
            "photoperiod_h": photoperiod,
            "light_drive": light_drive,
            "weight_drive": weight_drive,
            "hpg_signal": hpg_signal,
            "gnrh_index": gnrh,
            "lh_index": lh,
            "fsh_index": fsh,
            "estradiol_index": estradiol,
            "ovarian_state_index": ovarian_state,
        }
    )


def calculate_lay_metrics(
    df: pd.DataFrame,
    photostim_age_days: int,
    stress_index: float,
    egg_weight_g: float,
) -> pd.DataFrame:
    """Convert hormone activation into age at first egg and lay performance."""
    activation = df["hpg_signal"].to_numpy()
    days = df["day"].to_numpy()

    onset_threshold = 0.63
    onset_candidates = days[activation >= onset_threshold]
    age_first_egg = int(onset_candidates[0]) if len(onset_candidates) else int(days[-1] + 1)

    lag_penalty = max(0, 140 - photostim_age_days) * 0.12
    maturity_delay = max(0, 0.58 - activation.max()) * 28
    adjusted_age_first_egg = int(np.clip(age_first_egg + lag_penalty + maturity_delay, 120, 220))

    post_onset_days = np.maximum(days - adjusted_age_first_egg, 0)
    peak_curve = logistic(post_onset_days, midpoint=21, steepness=0.16)
    plateau_softener = 1.0 - 0.0009 * np.maximum(days - (adjusted_age_first_egg + 75), 0)
    plateau_softener = np.clip(plateau_softener, 0.82, 1.0)
    stress_modifier = np.clip(1.0 - 0.30 * (stress_index / 100.0), 0.7, 1.0)

    hen_day_production = 100 * peak_curve * plateau_softener * stress_modifier
    hen_day_production = np.where(days < adjusted_age_first_egg, 0.0, hen_day_production)
    cumulative_eggs = np.cumsum(hen_day_production / 100.0)

    egg_mass_g = cumulative_eggs * egg_weight_g
    feed_intake_g = 72 + 0.028 * df["body_weight_g"] + 0.92 * (hen_day_production / 10.0)
    feed_intake_g *= 1.0 + 0.08 * (stress_index / 100.0)

    output = df.copy()
    output["hen_day_production_pct"] = hen_day_production
    output["cumulative_eggs"] = cumulative_eggs
    output["egg_mass_g"] = egg_mass_g
    output["daily_feed_intake_g"] = feed_intake_g
    output["age_first_egg"] = adjusted_age_first_egg
    return output


def calculate_welfare_and_risks(df: pd.DataFrame, stress_index: float, photoperiod_after: float) -> dict:
    """Estimate simple risk and welfare outputs for teaching purposes."""
    final_bw = float(df["body_weight_g"].iloc[-1])
    peak_prod = float(df["hen_day_production_pct"].max())
    activation_peak = float(df["hpg_signal"].max())

    chronic_long_day_penalty = max(0.0, photoperiod_after - 16.0) * 2.5
    underweight_penalty = max(0.0, 1500.0 - final_bw) / 20.0
    overstimulation_penalty = max(0.0, activation_peak - 0.9) * 20.0

    welfare_score = 100 - (0.45 * stress_index) - chronic_long_day_penalty - underweight_penalty - overstimulation_penalty
    welfare_score = float(np.clip(welfare_score, 35, 100))

    delayed_maturity_risk = float(
        np.clip(5 + 0.18 * stress_index + max(0, 147 - int(df["age_first_egg"].iloc[0])) * 0.08, 0, 40)
    )
    erratic_lay_risk = float(np.clip(4 + 0.22 * stress_index + max(0.0, 15.0 - peak_prod / 6.0), 0, 35))
    metabolic_risk = float(np.clip(3 + chronic_long_day_penalty * 0.8 + max(0.0, peak_prod - 90.0) * 0.25, 0, 30))

    return {
        "welfare_score": welfare_score,
        "delayed_maturity_risk": delayed_maturity_risk,
        "erratic_lay_risk": erratic_lay_risk,
        "metabolic_risk": metabolic_risk,
    }


def summarize_outputs(df: pd.DataFrame, risks: dict) -> dict:
    """Collect dashboard summary metrics."""
    final = df.iloc[-1]
    peak_idx = df["hen_day_production_pct"].idxmax()
    peak = df.loc[peak_idx]

    return {
        "age_first_egg": int(df["age_first_egg"].iloc[0]),
        "final_body_weight_g": float(final["body_weight_g"]),
        "peak_hen_day_pct": float(peak["hen_day_production_pct"]),
        "peak_day": int(peak["day"]),
        "cumulative_eggs": float(final["cumulative_eggs"]),
        "final_hpg_signal": float(final["hpg_signal"]),
        "final_estradiol": float(final["estradiol_index"]),
        **risks,
    }


def generate_interpretation(summary: dict, photostim_age_days: int, photoperiod_after: float, stress_index: float) -> str:
    """Generate a classroom-friendly interpretation of the scenario."""
    tradeoffs = []

    if summary["age_first_egg"] < 150:
        tradeoffs.append("Photostimulation produced an early reproductive response, which can speed onset of lay.")
    elif summary["age_first_egg"] > 165:
        tradeoffs.append("The flock showed delayed sexual maturity, suggesting that light or body-weight readiness was not strong enough.")

    if photoperiod_after >= 16:
        tradeoffs.append("A longer day length strengthened HPG signaling, but sustained long photoperiods can trade some welfare margin for faster activation.")
    elif photoperiod_after <= 13:
        tradeoffs.append("A conservative photoperiod kept stimulation modest, which supports caution but can delay ovarian activation.")

    if photostim_age_days < 140:
        tradeoffs.append("Early photostimulation may challenge birds that are not fully body-weight ready.")
    elif photostim_age_days > 154:
        tradeoffs.append("Later photostimulation improved readiness but delayed time to first egg.")

    if stress_index >= 50:
        tradeoffs.append("High background stress suppressed hormone signaling and pulled down lay potential.")
    elif stress_index <= 20:
        tradeoffs.append("Low background stress allowed more of the light signal to translate into reproductive output.")

    if summary["welfare_score"] < 65:
        tradeoffs.append("This scenario creates a welfare caution, so students should discuss whether faster activation is worth the added strain.")

    return " ".join(tradeoffs[:4])


def make_curve_plot(df: pd.DataFrame) -> go.Figure:
    """Plot growth, HPG activation, and lay response over time."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["body_weight_g"],
            name="Body weight (g)",
            line=dict(color="#2F5D50", width=3),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["hpg_signal"] * 100,
            name="HPG activation (%)",
            line=dict(color="#D97B29", width=3),
            yaxis="y2",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["hen_day_production_pct"],
            name="Hen-day production (%)",
            line=dict(color="#A63D40", width=3),
            yaxis="y2",
        )
    )
    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Age (days)"),
        yaxis=dict(title="Body weight (g)", titlefont=dict(color="#2F5D50")),
        yaxis2=dict(
            title="Activation / production (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        legend=dict(orientation="h", y=1.08),
    )
    return fig


def make_hormone_plot(df: pd.DataFrame) -> go.Figure:
    """Plot hormone indices through time."""
    long_df = df.melt(
        id_vars="day",
        value_vars=["gnrh_index", "lh_index", "fsh_index", "estradiol_index"],
        var_name="signal",
        value_name="value",
    )
    label_map = {
        "gnrh_index": "GnRH",
        "lh_index": "LH",
        "fsh_index": "FSH",
        "estradiol_index": "Estradiol",
    }
    long_df["signal"] = long_df["signal"].map(label_map)
    return px.line(
        long_df,
        x="day",
        y="value",
        color="signal",
        height=420,
        color_discrete_map={
            "GnRH": "#375E97",
            "LH": "#FB6542",
            "FSH": "#3F681C",
            "Estradiol": "#9B59B6",
        },
        labels={"day": "Age (days)", "value": "Hormone index"},
    )


def make_risk_plot(risks: dict) -> go.Figure:
    """Plot simplified reproductive risk outcomes."""
    risk_df = pd.DataFrame(
        {
            "Outcome": ["Delayed maturity", "Erratic lay", "Metabolic strain"],
            "Risk (%)": [
                risks["delayed_maturity_risk"],
                risks["erratic_lay_risk"],
                risks["metabolic_risk"],
            ],
        }
    )
    return px.bar(
        risk_df,
        x="Risk (%)",
        y="Outcome",
        orientation="h",
        height=320,
        color="Risk (%)",
        color_continuous_scale=["#A7D49B", "#F1C453", "#C84C4C"],
        range_x=[0, 40],
    )


def run_simulation(
    hatch_weight_g: float,
    mature_weight_g: float,
    target_weight_pct: float,
    photostim_age_days: int,
    photoperiod_before: float,
    photoperiod_after: float,
    light_intensity_lux: float,
    spectrum_factor: float,
    stress_index: float,
    egg_weight_g: float,
) -> tuple[pd.DataFrame, dict]:
    """Run the full teaching simulation."""
    days = np.arange(0, SIM_DAYS + 1)
    body_weight = build_growth_curve(days, hatch_weight_g, mature_weight_g)
    body_weight *= np.clip(target_weight_pct / 100.0, 0.88, 1.08)

    df = calculate_hpg_activation(
        days=days,
        body_weight_g=body_weight,
        target_weight_pct=target_weight_pct,
        photostim_age_days=photostim_age_days,
        photoperiod_before=photoperiod_before,
        photoperiod_after=photoperiod_after,
        intensity_lux=light_intensity_lux,
        spectrum_factor=spectrum_factor,
        stress_index=stress_index,
    )
    df = calculate_lay_metrics(
        df=df,
        photostim_age_days=photostim_age_days,
        stress_index=stress_index,
        egg_weight_g=egg_weight_g,
    )
    risks = calculate_welfare_and_risks(df, stress_index, photoperiod_after)
    summary = summarize_outputs(df, risks)
    return df, summary


st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    .stMetric {background: #f7f3ea; border: 1px solid #ddd3c0; padding: 0.8rem; border-radius: 14px;}
    .hero {
        background: linear-gradient(135deg, #f3eadb 0%, #f8f6ef 45%, #e7f0e7 100%);
        border: 1px solid #d9d0bf;
        border-radius: 20px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1rem;
    }
    .caption-card {
        background: #fffdf8;
        border-left: 5px solid #b5792e;
        padding: 0.9rem 1rem;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:0.2rem;">Avian HPG Axis and Photostimulation Simulator</h1>
        <p style="margin-bottom:0.4rem;">
            Explore how body-weight readiness, light schedule, intensity, and background stress shape
            hypothalamic-pituitary-gonadal activation and the onset of lay.
        </p>
        <p style="margin:0; font-size:0.95rem;">
            Educational note: this is a biologically informed teaching model, not a validated commercial
            prediction system or a substitute for breeder guide, nutritionist, or veterinarian recommendations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Scenario Controls")

    genotype = st.selectbox("Bird type", ["Commercial layer pullet", "Broiler breeder pullet"])
    if genotype == "Commercial layer pullet":
        hatch_weight_g = 40.0
        mature_weight_default = 1550.0
        egg_weight_default = 60.0
    else:
        hatch_weight_g = 42.0
        mature_weight_default = 1950.0
        egg_weight_default = 64.0

    mature_weight_g = st.slider("Expected mature body weight (g)", 1400, 2200, int(mature_weight_default), 25)
    target_weight_pct = st.slider("Body weight vs. target at photostimulation (%)", 85, 110, 100, 1)
    photostim_age_days = st.slider("Age at photostimulation (days)", 119, 175, BASELINE_PHOTOSTIM_AGE, 1)

    st.subheader("Lighting")
    photoperiod_before = st.slider("Photoperiod before photostim (hours)", 8.0, 12.5, 10.0, 0.5)
    photoperiod_after = st.slider("Photoperiod after photostim (hours)", 12.5, 18.0, 15.0, 0.5)
    light_intensity_lux = st.slider("Light intensity (lux)", 1, 80, 20, 1)
    spectrum = st.selectbox("Light spectrum quality", ["Warm/red enriched", "Neutral white", "Poor spectrum"])
    spectrum_factor = {
        "Warm/red enriched": 1.05,
        "Neutral white": 1.0,
        "Poor spectrum": 0.9,
    }[spectrum]

    st.subheader("Management context")
    stress_index = st.slider("Background stress challenge", 0, 100, 20, 1)
    egg_weight_g = st.slider("Average egg weight after onset (g)", 52, 72, int(egg_weight_default), 1)


df, summary = run_simulation(
    hatch_weight_g=hatch_weight_g,
    mature_weight_g=float(mature_weight_g),
    target_weight_pct=float(target_weight_pct),
    photostim_age_days=int(photostim_age_days),
    photoperiod_before=float(photoperiod_before),
    photoperiod_after=float(photoperiod_after),
    light_intensity_lux=float(light_intensity_lux),
    spectrum_factor=float(spectrum_factor),
    stress_index=float(stress_index),
    egg_weight_g=float(egg_weight_g),
)

interpretation = generate_interpretation(summary, photostim_age_days, photoperiod_after, stress_index)

metric_cols = st.columns(6)
metric_cols[0].metric("Age at first egg", f"{summary['age_first_egg']} d")
metric_cols[1].metric("Peak hen-day production", f"{summary['peak_hen_day_pct']:.1f}%")
metric_cols[2].metric("Peak production day", f"{summary['peak_day']}")
metric_cols[3].metric("Final BW", f"{summary['final_body_weight_g']:.0f} g")
metric_cols[4].metric("Cumulative eggs", f"{summary['cumulative_eggs']:.1f}")
metric_cols[5].metric("Welfare score", f"{summary['welfare_score']:.0f}/100")

left, right = st.columns([1.45, 1.0])

with left:
    st.subheader("Growth, Activation, and Lay")
    st.plotly_chart(make_curve_plot(df), use_container_width=True)

with right:
    st.subheader("Scenario Interpretation")
    st.markdown(f'<div class="caption-card">{interpretation}</div>', unsafe_allow_html=True)

    if summary["welfare_score"] < 65 or summary["metabolic_risk"] > 18:
        st.warning(
            "This scenario pushes reproductive activation with a narrower welfare margin. "
            "Discuss whether the timing, day length, or stress load should be moderated."
        )
    if summary["age_first_egg"] > 165:
        st.info(
            "Delayed maturity suggests that birds may not have been sufficiently ready in body weight, "
            "light intensity, or photostimulatory day length."
        )

    risk_cols = st.columns(2)
    risk_cols[0].metric("Delayed maturity risk", f"{summary['delayed_maturity_risk']:.1f}%")
    risk_cols[1].metric("Metabolic strain risk", f"{summary['metabolic_risk']:.1f}%")


bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.subheader("Hormone Indices")
    st.plotly_chart(make_hormone_plot(df), use_container_width=True)

with bottom_right:
    st.subheader("Risk Overview")
    st.plotly_chart(make_risk_plot(summary), use_container_width=True)


st.subheader("Daily Output Table")
display_df = df[
    [
        "day",
        "body_weight_g",
        "photoperiod_h",
        "hpg_signal",
        "gnrh_index",
        "lh_index",
        "fsh_index",
        "estradiol_index",
        "hen_day_production_pct",
        "cumulative_eggs",
        "daily_feed_intake_g",
    ]
].copy()
display_df.columns = [
    "Day",
    "Body weight (g)",
    "Photoperiod (h)",
    "HPG activation",
    "GnRH index",
    "LH index",
    "FSH index",
    "Estradiol index",
    "Hen-day production (%)",
    "Cumulative eggs",
    "Daily feed intake (g)",
]
st.dataframe(display_df.round(2), use_container_width=True, height=320)

csv = display_df.to_csv(index=False).encode("utf-8")
st.download_button("Download daily results as CSV", csv, "avian_hpg_photostim_simulation.csv", "text/csv")

with st.expander("Instructor Mode: Model Logic"):
    st.markdown(
        """
        - `Body weight` follows a logistic pullet growth curve scaled by the selected mature size and target-weight achievement.
        - `Light stimulation` rises with longer day length, stronger light intensity, and better spectrum quality.
        - `HPG activation` depends on age readiness, body-weight readiness, light stimulation, and a stress penalty.
        - `Hormone indices` are simple sigmoidal transformations of HPG activation and are displayed as teaching indices, not laboratory units.
        - `Age at first egg` occurs when activation passes a threshold, then is adjusted for very early photostimulation and incomplete maturity.
        - `Hen-day production` ramps up after first egg, reaches a peak, then gently plateaus.
        - `Risk outputs` and `welfare score` are simplified educational scores to support classroom discussion about tradeoffs.
        """
    )

st.caption(
    "This simulator is intended for classroom exploration of avian reproductive physiology. "
    "It simplifies real breeder and layer biology and should not be used as a commercial decision tool."
)
