# Quantum-Assisted Irrigation Optimization

A complete academic/demo implementation of:

**Irrigation dataset → constrained QUBO → QAOA with Qiskit → classical exact comparison → final irrigation plan**

## Project objective

The system receives plot-level stress and water requirements. It converts the irrigation decision into binary variables:

- `x_i = 1`: irrigate plot `i`
- `x_i = 0`: skip plot `i`

A water-budget constraint is converted into a QUBO using binary slack variables:

\[
\min_x -\sum_i B_i x_i +
P\left(\sum_i W_i x_i + \sum_j S_j y_j-W\right)^2
\]

where:

- `B_i` = benefit/stress score
- `W_i` = water required by plot i
- `W` = available water
- `y_j` = slack variables
- `P` = constraint penalty

The resulting QUBO is mapped to an Ising Hamiltonian and optimized with QAOA.

## Current Qiskit approach

This project uses the current Qiskit-style QAOA circuit API:

```python
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp, Statevector
```

It does **not** depend on the older `qiskit.algorithms.QAOA` interface.

## Folder structure

```text
quantum_irrigation_optimizer/
├── data/
│   └── irrigation_data.csv
├── src/
│   └── main.py
├── outputs/
│   └── generated after execution
├── requirements.txt
└── README.md
```

## 1. Install Python

Python 3.10+ is recommended.

## 2. Create an environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the project

From the project root:

```bash
python src/main.py
```

The default water budget is 60% of the total demand in the sample dataset.

You can set your own water budget:

```bash
python src/main.py --water-budget 140
```

Increase QAOA depth:

```bash
python src/main.py --water-budget 140 --reps 2 --maxiter 100
```

## 5. Generated files

The program creates:

- `outputs/qaoa_irrigation_plan.csv`
- `outputs/classical_exact_irrigation_plan.csv`
- `outputs/method_comparison.csv`
- `outputs/final_report.txt`
- `outputs/qaoa_irrigation_plan.png`
- `outputs/qaoa_vs_exact.png`

## Dataset format

Your CSV should contain at least:

```text
plot_id,crop,water_requirement_liters,stress_score
```

The included dataset has additional useful fields:

```text
plot_id
crop
soil_moisture_pct
moisture_threshold_pct
temperature_c
humidity_pct
crop_stage
priority
water_requirement_liters
stress_score
```

`stress_score` must be between 0 and 1.

Example:

```text
P01,Rice,31,55,33,48,Flowering,5,40,0.92
```

## Important: using a real dataset

The included CSV is a **demonstration dataset**, not a claim that these measurements came from a real farm.

For your final project, replace it with a real irrigation/soil-moisture dataset. Keep the required columns or modify `load_dataset()` to match your dataset.

If your real dataset does not have `stress_score`, we can add a separate machine-learning stage to predict stress from soil moisture, temperature, humidity, crop type, etc.

## How the optimization works

### Binary decision

```text
x_i = 1 → irrigate
x_i = 0 → skip
```

### Constraint

```text
total irrigation water <= available water
```

### QUBO

A binary slack representation converts the inequality into an equality:

```text
water used + slack = water budget
```

The penalty term makes allocations exceeding the water budget expensive.

### Classical exact solver

For this small academic example, every binary state is enumerated and the minimum QUBO value is found. This gives a reference solution.

### QAOA

The QUBO is converted to an Ising Hamiltonian using:

```text
x_i = (1 - Z_i) / 2
```

QAOA then searches for low-energy states using a parameterized quantum circuit and a classical optimizer.

## Interpretation

Do not claim that QAOA is faster or better than the exact solver from this small simulation.

The purpose of the project is to demonstrate:

1. Real/representative irrigation data
2. Binary optimization modeling
3. QUBO formulation
4. QUBO-to-Ising mapping
5. QAOA implementation
6. Classical exact benchmarking
7. Practical irrigation-plan output

## Optional extension: ML + QAOA

A stronger version can add:

```text
Historical farm data
        ↓
Machine learning model
        ↓
Predicted stress / water requirement
        ↓
QUBO
        ↓
QAOA
        ↓
Optimal irrigation plan
```

This turns the project into a prediction + optimization pipeline.
