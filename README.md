# CellScope

**Created by Tabish Syed**

CellScope is a battery-aging analysis workbench. It uses cycle-level test measurements to estimate remaining cycles to a chosen state-of-health threshold and flag unusual measurements for review.

The project demonstrates a complete small ML workflow: synthetic data generation, cell-level holdout evaluation, feature engineering, model inference, an interactive dashboard, and a reproducible local setup.

## Run locally

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Explore the demo

The default dataset contains simulated measurements from multiple battery cells across charge-discharge cycles. Choose a cell to inspect its capacity health, estimated remaining cycles, and anomaly flags. The evaluation panel reports performance on cells withheld entirely from model fitting, which avoids mixing cycles from the same cell across train and evaluation sets.

## Analyze a CSV

Upload cycle-level measurements with these columns:

| Column | Meaning |
| --- | --- |
| `cell_id` | Cell or sample identifier |
| `cycle` | Cycle number, increasing within each cell |
| `capacity_ah` | Measured discharge capacity in amp-hours |
| `nominal_capacity_ah` | Cell's nominal capacity in amp-hours |
| `resistance_mohm` | Measured internal resistance in milliohms |
| `temperature_c` | Temperature in Celsius |
| `voltage_sag_mv` | Voltage sag under a defined load, in millivolts |
| `charge_time_min` | Charge duration in minutes |

Each cell needs at least five complete cycle rows. The uploaded data is scored by a model trained on synthetic data; estimates should not be treated as calibrated for a real chemistry, cell design, or test protocol.

## Method

A gradient-boosting regressor predicts the number of cycles remaining until the simulated cell crosses 80% of its nominal capacity. Input features include state of health, cycle count, capacity trend, resistance, temperature, voltage sag, and charge time. Evaluation uses a grouped cell holdout.

An Isolation Forest flags feature combinations that differ from the training set. An anomaly flag is a prompt to inspect a test record; it does not diagnose a fault.

## Limits and safety

The included measurements are synthetic and exist to demonstrate modeling and evaluation. Battery degradation depends on chemistry, cell design, manufacturing, operating conditions, and test methods. This project is not validated for real cells and must not guide charging, maintenance, warranty, safety, or end-of-life decisions.

## License

MIT. See [LICENSE](LICENSE).
