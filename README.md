# Time-series
This project implements an advanced deep learning pipeline for probabilistic time series forecasting using LSTM or Temporal Convolutional Networks (TCN).
The goal is not only accurate point forecasting but also robust uncertainty quantification, enabling data-driven decision-making under uncertainty.

The model outputs:

Point forecast (mean)

Aleatoric uncertainty (data noise)

Epistemic uncertainty (model uncertainty) via Monte Carlo Dropout

Combined prediction intervals (80% and 95%)

The project includes:

Synthetic real-world–like dataset generation

Neural network-based probabilistic forecasting

Evaluation of forecast accuracy and uncertainty calibration

Visualization of predictions with uncertainty bands

📌 Project Highlights

✔ Synthetic daily dataset (3 years) with trend, seasonalities, and heteroscedastic noise
✔ Feature engineering (cyclic time features, lag windows)
✔ LSTM model producing mean + log-variance trained via Gaussian Negative Log-Likelihood
✔ MC Dropout for uncertainty estimation
✔ Forecast metrics: RMSE, MAE
✔ Probabilistic metrics: Coverage Probability, Mean Interval Width
✔ Visualization of 80% & 95% Prediction Intervals
✔ Fully documented, PEP 8–compliant, production-ready code

📂 File Structure
project/
│
├── probabilistic_lstm_forecast.py   # Main runnable pipeline
├── ts_forecast_outputs/             # Generated metrics + plots
│   ├── metrics.csv
│   ├── forecast_plot_*.png
│
└── README.md                        # This file

📊 Dataset Description

The dataset is synthetically generated to emulate a real-world time series with:

Linear trend

Yearly seasonality

Weekly seasonality

Heteroscedastic noise (time-varying variance)

Random outliers

This ensures a challenging forecasting task suitable for evaluating uncertainty-aware models.

🧠 Model Architecture

The forecasting model is an LSTM-based probabilistic neural network:

Inputs

30-day lag window

Cyclic features:

sin_doy, cos_doy (day-of-year)

sin_dow, cos_dow (day-of-week)

Outputs

μ → mean forecast

log(σ²) → predicted variance

Loss

Gaussian Negative Log Likelihood (NLL)

Uncertainty Estimation

Aleatoric uncertainty → learned directly from log_var

Epistemic uncertainty → Monte Carlo Dropout inference

Total predictive variance = aleatoric + epistemic

📈 Evaluation Metrics
Point Forecast Metrics
Metric	Description
RMSE	Root Mean Squared Error
MAE	Mean Absolute Error
Uncertainty / Probabilistic Metrics
Metric	Meaning
Coverage_80	Fraction of true values inside 80% PI
Mean_Width_80	Average width of 80% PI
Coverage_95	Calibration accuracy of 95% PI
Mean_Width_95	Sharpness vs uncertainty trade-off
📦 Requirements

Install required libraries:

pip install numpy pandas scikit-learn matplotlib tensorflow scipy


To run CPU-only TensorFlow:

pip install tensorflow-cpu

🚀 How to Run
1. Clone the repository
git clone <your-repo-url>
cd project/

2. Run the script
python probabilistic_lstm_forecast.py

3. View Outputs

Generated files will be saved in:

ts_forecast_outputs/


Includes:

metrics.csv → evaluation metrics

forecast_plot_*.png → forecast vs actual with 80/95% intervals

📊 Example Output (when you run the script)
rmse: 0.874321
mae: 0.623145
coverage_80: 0.79
coverage_95: 0.95
mean_width_80: 1.20
mean_width_95: 2.45


(Values vary depending on training)

🧪 Hyperparameter Tuning

Recommended grid options:

LSTM units: 32, 64, 128

Dropout rate: 0.1–0.3

Lookback window: 14, 30, 60

Learning rate: 1e-3, 5e-4

Optimizer: Adam

Time-series cross-validation via rolling window is advised.

📘 Future Work

🔹 Add TCN-based probabilistic forecaster
🔹 Implement quantile regression using Pinball Loss
🔹 Add multistep (H>1) forecasting support
🔹 Compare MC Dropout vs Deep Ensembles
🔹 Extend synthetic dataset to include:

regime shifts

structural breaks

missing data patterns

🤝 Contributing

Pull requests and improvements are welcome.
Follow PEP 8 and maintain docstring documentation for all functions and modules.
