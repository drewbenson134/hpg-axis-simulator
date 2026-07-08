# Avian HPG Axis and Photostimulation Simulator

This project is a Streamlit teaching app that demonstrates how body-weight readiness, light schedule, light intensity, and background stress can influence avian hypothalamic-pituitary-gonadal (HPG) activation and the onset of lay.

## Educational purpose

This is a simplified educational simulator. It is not a validated prediction model and should not replace breeder guides, field data, nutrition advice, or veterinary recommendations.

## Features

- Interactive scenario controls in the sidebar
- Simulated pullet growth and body-weight readiness
- Photostimulation timing and light-program adjustments
- HPG activation and hormone index curves
- Estimated age at first egg and hen-day production response
- Simple welfare and reproductive risk indicators
- Downloadable daily output table as CSV
- Instructor mode with transparent model logic

## Files

- `app.py`: main Streamlit app
- `requirements.txt`: Python dependencies

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Modeling notes

- Growth is represented with a logistic pullet growth curve.
- HPG activation is driven by age, body-weight readiness, photoperiod, light intensity, spectrum quality, and background stress.
- Hormone outputs are presented as teaching indices rather than measured lab values.
- Lay onset occurs when activation reaches a threshold and is then adjusted for timing and readiness effects.
- Welfare and risk outputs are simplified classroom discussion metrics.
