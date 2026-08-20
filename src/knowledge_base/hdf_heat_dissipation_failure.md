# Heat Dissipation Failure (HDF)

## Failure Mode Overview
Heat Dissipation Failure (HDF) occurs when the machine cannot adequately dissipate the thermal energy generated during the cutting and machining process, causing critical overheating at the tool-workpiece interface and spindle bearing assembly.

## Mechanism
Heat dissipation in the machine is driven by convection and cooling air flow, which is directly coupled with the spindle's rotational speed (fan/airflow dynamics) and the temperature gradient between the process chamber and ambient surroundings. If the spindle speed is too low to maintain sufficient cooling airflow, and the temperature difference between the process and ambient air is too small to drive adequate thermal convection, heat accumulates rapidly, causing thermal expansion, lubricant degradation, and thermal seizure.

## Trigger Condition (Ground Truth AI4I 2020 Formula)
In the AI4I 2020 dataset, HDF triggers if and only if both of the following physical criteria are simultaneously satisfied:
1. `(Process temperature [K] - Air temperature [K]) < 8.6 K` (insufficient convective thermal gradient)
2. `Rotational speed [rpm] < 1380 rpm` (insufficient cooling airflow)

Formula: `(Process temperature - Air temperature < 8.6 K) AND (Rotational speed < 1380 rpm)`.

## What to Look For
- Sensor telemetry displaying `Process temperature [K] - Air temperature [K] < 8.6 K` (e.g. Temp_Diff of 7.5 K - 8.5 K).
- Spindle `Rotational speed [rpm] < 1380 rpm`.
- SHAP feature attributions showing `Temp_Diff`, `Process temperature`, `Air temperature`, or `Rotational speed` significantly driving failure probability.

## Recommended Maintenance Action
1. Inspect the machine's thermal cooling circuit, heat exchangers, cooling fans, and ventilation ducts.
2. Clean or replace clogged air/oil filtration units (Part Category: `cooling_system` or `air_filter`).
3. Verify coolant fluid flow rates, nozzle alignment, and thermal sensor calibration.
4. Check whether operating duty cycle allows adequate dwell time between heavy machining operations.

## Distinguishing From Other Failure Modes
- **vs. PWF (Power Failure)**: PWF occurs when instantaneous power ($P = \tau \times \omega$) exceeds 9000 W or drops below 3500 W. HDF specifically requires low RPM (< 1380) coupled with a low temperature gradient (< 8.6 K).
- **vs. OSF (Overstrain Failure)**: OSF is governed by the product of Tool Wear and Torque ($ToolWear \times Torque > \text{limit}$), having no direct dependence on temperature differentials.
- **vs. TWF (Tool Wear Failure)**: TWF is strictly time-in-cut wear (200–240 min), whereas HDF can occur with a brand-new tool (0 min wear) if thermal conditions degrade.
- **vs. RNF (Random Failure)**: RNF has zero sensor correlation, whereas HDF is strictly governed by the 8.6 K differential and 1380 rpm boundary.
