# Tool Wear Failure (TWF)

## Failure Mode Overview
Tool Wear Failure (TWF) represents the gradual, progressive mechanical degradation and wear of the cutting tool insert or edge across sustained machine machining cycles.

## Mechanism
As the tool engages with workpieces across consecutive cycles, continuous friction, abrasive forces, adhesive wear, and thermal fatigue erode the cutting edge geometry. Beyond a critical wear envelope, the cutting edge loses structural sharpness and micro-fractures, leading to catastrophic tool failure or severe workpiece damage.

## Trigger Condition (Ground Truth AI4I 2020 Formula)
In the AI4I 2020 industrial setup:
- The tool wear accumulates progressively with each machining operation.
- Once cumulative `Tool wear [min]` enters the critical threshold window between **200 minutes and 240 minutes** (mean failure probability centered around ~220 minutes), the tool will fail randomly (stochastically) at some point within this range.
- Specifically: `200 <= Tool wear [min] <= 240`.

## What to Look For
- Sensor telemetry displaying `Tool wear [min] >= 200 min` (or approaching 200–240 min).
- Progressive upward slope in tool wear over recent temporal readings.
- Accompanying minor increases in cutting torque or vibration as a dulling tool meets higher resistance.
- SHAP attribution indicating `Tool wear` is the primary positive driver increasing failure risk.

## Recommended Maintenance Action
1. Schedule a tool insert replacement before entering or continuing operations inside the 200–240 min danger zone.
2. Inspect the tool holder, spindle chuck, and collet for micro-damage or runout.
3. Check spare parts inventory for replacement cutting tool inserts (Part Category: `tool_insert`).
4. Upon physical replacement, reset the machine's `Tool wear [min]` timer to 0.

## Distinguishing From Other Failure Modes
- **vs. OSF (Overstrain Failure)**: OSF requires the product of `Tool wear [min]` and `Torque [Nm]` to exceed variant limits (11,000 / 12,000 / 13,000). TWF can trigger even under normal or low torque conditions if tool wear time reaches 200–240 min.
- **vs. PWF (Power Failure)**: PWF is driven purely by electrical/mechanical instantaneous power (`Torque * RPM * 2*pi/60` < 3500 W or > 9000 W) regardless of tool wear.
- **vs. HDF (Heat Dissipation Failure)**: HDF depends strictly on low temperature differential (`Process Temp - Air Temp < 8.6 K`) and low rotational speed (`RPM < 1380`).
- **vs. RNF (Random Failure)**: RNF is a flat 0.1% background failure uncorrelated with any physical feature, whereas TWF is strongly correlated with tool wear duration >= 200 min.
