# anomaly_detection_module.py

import numpy as np
from sklearn.ensemble import IsolationForest  # Assuming you're using Isolation Forest for anomaly detection

def detect_anomalies(financial_data):
    # Assuming financial_data is a 2D array or DataFrame with financial transaction data
    # You can replace this with your actual anomaly detection logic

    # Example using Isolation Forest
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    predictions = isolation_forest.fit_predict(financial_data)
    
    # Anomaly scores (the lower, the more anomalous)
    anomaly_scores = isolation_forest.decision_function(financial_data)

    return anomaly_scores, predictions
