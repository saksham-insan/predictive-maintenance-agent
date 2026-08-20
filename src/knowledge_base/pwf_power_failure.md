# Power Failure (PWF)

## Failure Mode Overview
Power Failure (PWF) represents an instantaneous mechanical/electrical power overload or underload condition where the required or delivered spindle drive power falls outside safe operating boundaries.

## Mechanism
Mechanical power consumed during rotation is the product of applied torque and rotational velocity:
$$P = \tau \times \omega = \text{Torque [Nm]} \times \left( \text{Rotational speed [rpm]} \times \frac{2\pi}{60} \right) \approx \text{Torque} \times \text{RPM} \times 0.10472 \text{ [Watts]}$$
If the required cutting power exceeds the rated drive motor continuous/peak rating (> 9000 W), the electrical drive inverter trips, current spikes, and motor windings overheat. Conversely, if power drops below 3500 W during an active machining cycle, it signals motor stalling, slip, or extreme underload.

## Trigger Condition (Ground Truth AI4I 2020 Formula)
In the AI4I 2020 dataset:
$$\text{Power [W]} = \text{Torque [Nm]} \times \text{Rotational speed [rpm]} \times \frac{2\pi}{60}$$
PWF occurs if:
- $\text{Power [W]} < 3500\text{ W}$, OR
- $\text{Power [W]} > 9000\text{ W}$

## What to Look For
- Calculated instantaneous power ($Torque \times RPM \times \frac{2\pi}{60}$) outside the $[3500\text{ W}, 9000\text{ W}]$ normal operating band.
- High torque combined with moderate-to-high RPM (e.g. Torque = 65 Nm at 1450 RPM gives ~9870 W > 9000 W).
- SHAP feature attributions with `Power`, `Torque`, or `Rotational speed` having dominant positive impact on failure risk.

## Recommended Maintenance Action
1. Inspect electrical drive inverter, motor power cables, and spindle motor bearings.
2. Verify load balance, power supply voltage stability, and spindle VFD (variable frequency drive) parameter tuning (Part Category: `drive_motor` or `power_inverter`).
3. Optimize CNC feed rate and cutting depths to keep machining power within the 3500 W – 8500 W safe window.
4. Check for mechanical binding or spindle axis misalignment causing excessive resistive torque.

## Distinguishing From Other Failure Modes
- **vs. OSF (Overstrain Failure)**: OSF requires the product of `Tool wear` and `Torque` to exceed 11,000–13,000 min·Nm. PWF depends on `Torque` and `Rotational speed` ($Torque \times RPM$), regardless of whether tool wear is 0 min or 200 min.
- **vs. HDF (Heat Dissipation Failure)**: HDF depends on $(ProcTemp - AirTemp) < 8.6\text{ K}$ and $RPM < 1380$, whereas PWF is purely an instantaneous power equation.
- **vs. TWF (Tool Wear Failure)**: TWF requires tool wear >= 200 min, whereas PWF can occur at any tool wear level.
- **vs. RNF (Random Failure)**: RNF is flat 0.1% background noise, while PWF strictly correlates with Power < 3500 W or > 9000 W.
