# Random Failure (RNF)

## Failure Mode Overview
Random Failure (RNF) represents unmodeled, anomalous stochastic failures caused by unforeseen external events, unpredictable material impurities, or minor unmonitored sub-system disturbances.

## Mechanism
Unlike the physical degradation mechanisms (TWF, HDF, PWF, OSF), Random Failures in industrial operations represent the intrinsic baseline failure rate that cannot be traced to any measured sensor variable or operating parameter. They are completely uncorrelated with machine load, speed, temperature, or wear.

## Trigger Condition (Ground Truth AI4I 2020 Formula)
In the AI4I 2020 dataset:
- RNF occurs with a flat, constant **0.1% probability** (1 in 1,000 operations) uniformly across all machine types and operating conditions.
- It has **zero mathematical or physical correlation** with temperature, torque, speed, or tool wear.

## What to Look For
- Anomaly flagged or subtle borderline signal where none of the physical trigger conditions for TWF, HDF, PWF, or OSF are met:
  - Tool wear is well below 200 min (e.g. 50 min).
  - Temperature difference is healthy (> 8.6 K) and RPM is normal (> 1380 rpm).
  - Instantaneous power is comfortably within the 3500 W – 9000 W operating window.
  - Tool wear × Torque is well below 11,000 min·Nm.
- SHAP feature attributions show low or dispersed values across features without a single clear culprit.

## Recommended Maintenance Action
1. Conduct a non-destructive functional test and visual safety check before taking drastic action.
2. If uncertain, call `request_more_sensor_data` to gather subsequent telemetry before replacing healthy parts.
3. Do NOT prematurely replace expensive parts (such as spindle motors or fresh tool inserts) when no physical sensor threshold is breached.
4. If erratic signals persist, escalate to a maintenance engineer for electrical insulation and calibration inspection.

## Distinguishing From Other Failure Modes
- **vs. TWF, HDF, PWF, OSF**: All other failure modes have exact, deterministic physical formulas and unmistakable sensor signatures. If an anomaly occurs without meeting any of the physical formulas for TWF (wear 200–240 min), HDF (delta-T < 8.6 K & RPM < 1380), PWF (power < 3500 W or > 9000 W), or OSF (wear * torque > 11k/12k/13k), it is likely a low-signal stochastic event (RNF) or benign transient fluctuation.
