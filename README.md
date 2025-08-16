# Predictive Maintenance for Industrial Machines

## Project Overview
This project predicts the **Remaining Useful Life (RUL)** of industrial equipment using historical sensor data.  
It applies deep learning techniques to monitor and forecast failures in motors, generators, transformers, and turbines which are essential assets in Electrical Engineering applications. Early predictions help reduce downtime, improve safety, and avoid electrical hazards.

---

## Electrical Engineering Connection
Many EE roles in power plants, factories, and grid systems involve ensuring reliable operation of electrical equipment.  
This project directly applies to monitoring electrical machines, predicting failures, and optimizing maintenance schedules.

---

## Machine Learning Approach
**Supervised Learning** problem:
- **Labels:** Remaining life in cycles/hours until failure.  
- **Features:** 21 Sensor readings (vibration, temperature, pressure, current, etc.).  
- **Algorithm Base:** Regression → extended to deep learning regression (LSTM for time series).

---

## Dataset
**NASA CMAPSS Turbofan Engine Degradation Simulation Data**  

- Public, well-documented dataset simulating multiple engines under different conditions until failure.  
- Each time step contains 21 sensor readings and an engine ID.  
---

## Step-by-Step Implementation

### 1. Preprocessing
- Convert the TXT file to CSV (better visualize and for further processing)
- Scale sensor values.

### 2. Modeling
- **Deep Learning:**  
  - Fully Connected Neural Network for tabular data.  
  - LSTM for time-series patterns.  
- **Loss Function:** Mean Squared Error (MSE).

### 3. Evaluation 
- Cross-validation across engines to test generalization.

### 4. Deployment
- Streamlit web app:  
  - Upload recent sensor logs (TXT).  
  - Predict RUL. 
