# Overstrain Failure (OSF)

## Failure Mode Overview
Overstrain Failure (OSF) occurs when high cutting torque interacts with advanced tool wear, exceeding the structural mechanical strain limit of the machine tool and workpiece interface.

## Mechanism
As a cutting tool wears, contact friction and required cutting forces increase substantially. When high torque is demanded against a worn cutting edge, the product of mechanical force and accumulated micro-wear exceeds the yield strength of the tool assembly or workpiece fixture, causing sudden overstrain rupture, plastic deformation, or catastrophic tool breakage.

## Trigger Condition (Ground Truth AI4I 2020 Formula)
In the AI4I 2020 dataset, OSF triggers when the product of `Tool wear [min]` and `Torque [Nm]` exceeds a variant-specific overstrain threshold:
- **Type L (Low product variant)**: $\text{Tool wear [min]} \times \text{Torque [Nm]} > 11,000 \text{ min}\cdot\text{Nm}$
- **Type M (Medium product variant)**: $\text{Tool wear [min]} \times \text{Torque [Nm]} > 12,000 \text{ min}\cdot\text{Nm}$
- **Type H (High product variant)**: $\text{Tool wear [min]} \times \text{Torque [Nm]} > 13,000 \text{ min}\cdot\text{Nm}$

## What to Look For
- Combination of elevated `Tool wear [min]` and high `Torque [Nm]` whose product exceeds the product variant's threshold (11k for L, 12k for M, 13k for H).
- Example: Type M machine with `Tool wear = 215 min` and `Torque = 58 Nm` gives $215 \times 58 = 12,470 > 12,000$, which triggers OSF.
- SHAP feature attributions showing both `Torque` and `Tool wear` (and `Type_Encoded`) heavily contributing to failure risk.

## Recommended Maintenance Action
1. Immediately reduce feed rate and cutting depth on the active CNC program to alleviate mechanical torque load.
2. Replace the worn cutting tool insert (Part Category: `tool_insert`).
3. Inspect spindle shaft, collet chuck, and machine guide rails for deflection or mechanical strain damage.
4. Verify work-holding fixture rigidity and workpiece material hardness consistency.

## Distinguishing From Other Failure Modes
- **vs. TWF (Tool Wear Failure)**: TWF is time-dependent (200–240 min) and can occur even with light torque. OSF can occur earlier (e.g. at 180 min tool wear with 70 Nm torque on Type L: $180 \times 70 = 12,600 > 11,000$).
- **vs. PWF (Power Failure)**: PWF depends on $Torque \times RPM$ (power), regardless of tool wear. OSF depends on $Tool Wear \times Torque$.
- **vs. HDF (Heat Dissipation Failure)**: HDF depends on $(ProcTemp - AirTemp) < 8.6\text{ K}$ and $RPM < 1380$, whereas OSF is a mechanical force-wear interaction.
- **vs. RNF (Random Failure)**: RNF is flat 0.1% probability with no correlation to torque or wear, whereas OSF is strictly deterministic based on the $11\text{k}/12\text{k}/13\text{k}$ product threshold.
