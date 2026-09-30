
"""
Quantum-Assisted Irrigation Optimization
-----------------------------------------
Pipeline:
1. Load irrigation dataset.
2. Build a constrained QUBO using binary irrigation variables and slack variables.
3. Solve the QUBO exactly by enumeration.
4. Convert the QUBO to an Ising Hamiltonian.
5. Run QAOA with Qiskit's qaoa_ansatz + Statevector simulator.
6. Sample candidate bitstrings and select the best feasible QAOA solution.
7. Compare QAOA with the classical exact solution.
8. Save a final irrigation plan and comparison plot.

The included CSV is a small demonstration dataset. Replace it with your own
dataset using the same required columns, or modify load_dataset().
"""

# from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp, Statevector


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "irrigation_data.csv"
DEFAULT_OUTPUT = ROOT / "outputs"


REQUIRED_COLUMNS = {
    "plot_id",
    "crop",
    "water_requirement_liters",
    "stress_score",
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate the irrigation dataset."""
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["water_requirement_liters"] = pd.to_numeric(
        df["water_requirement_liters"], errors="raise"
    )
    df["stress_score"] = pd.to_numeric(df["stress_score"], errors="raise")

    if (df["water_requirement_liters"] <= 0).any():
        raise ValueError("All water requirements must be positive.")

    if ((df["stress_score"] < 0) | (df["stress_score"] > 1)).any():
        raise ValueError("stress_score must be between 0 and 1.")

    if df["plot_id"].duplicated().any():
        raise ValueError("plot_id values must be unique.")

    return df


def choose_default_capacity(df: pd.DataFrame, fraction: float = 0.60) -> int:
    """Choose a demonstration water budget from the dataset."""
    total = int(np.ceil(df["water_requirement_liters"].sum()))
    return max(1, int(np.floor(total * fraction)))


def build_qubo(
    df: pd.DataFrame,
    water_budget: int,
    penalty: float = 2.0,
):
    """
    Build a constrained QUBO.

    Plot variable x_i = 1 means plot i is irrigated.
    Slack variables make:
        sum(w_i*x_i) + sum(s_j*slack_j) = water_budget
    possible for every feasible allocation.

    Objective:
        minimize -sum(benefit_i*x_i)
                 + penalty*(sum(w_i*x_i)+sum(slack_j*s_j)-W)^2

    benefit_i combines stress and priority without requiring an ML model.
    """
    n_plots = len(df)
    weights = [int(v) for v in df["water_requirement_liters"]]

    # Slack bits 1,2,4,... are enough to represent all values up to W.
    n_slack = max(1, int(math.ceil(math.log2(water_budget + 1))))
    slack_weights = [2**j for j in range(n_slack)]

    all_weights = weights + slack_weights
    n_vars = len(all_weights)

    stress = df["stress_score"].to_numpy(dtype=float)

    if "priority" in df.columns:
        priority = pd.to_numeric(
            df["priority"], errors="coerce"
        ).fillna(1).to_numpy(dtype=float)
        pmax = max(float(priority.max()), 1.0)
        priority_norm = priority / pmax
        # Stress dominates; priority is a smaller tie-breaker.
        benefit = 0.85 * stress + 0.15 * priority_norm
    else:
        benefit = stress.copy()

    linear = np.zeros(n_vars, dtype=float)
    quadratic = {}

    # penalty*(A(x)-W)^2
    # Because x_i^2 = x_i for binary variables:
    # linear_i += P*(a_i^2 - 2W*a_i)
    for i, a in enumerate(all_weights):
        linear[i] = (-benefit[i] if i < n_plots else 0.0)
        linear[i] += penalty * (a * a - 2 * water_budget * a)

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            quadratic[(i, j)] = 2.0 * penalty * all_weights[i] * all_weights[j]

    constant = penalty * water_budget**2

    return {
        "linear": linear,
        "quadratic": quadratic,
        "constant": constant,
        "weights": all_weights,
        "plot_weights": weights,
        "slack_weights": slack_weights,
        "benefit": benefit,
        "n_plots": n_plots,
        "n_vars": n_vars,
        "water_budget": water_budget,
        "penalty": penalty,
    }


def qubo_value(bits, qubo) -> float:
    """Evaluate the QUBO objective for a binary vector."""
    bits = np.asarray(bits, dtype=int)
    value = float(qubo["constant"])
    value += float(np.dot(qubo["linear"], bits))

    for (i, j), coeff in qubo["quadratic"].items():
        value += coeff * bits[i] * bits[j]

    return float(value)


def decode_bits(bits, qubo):
    """Decode plot/slack variables from a binary vector."""
    bits = np.asarray(bits, dtype=int)
    n = qubo["n_plots"]
    plot_bits = bits[:n]
    slack_bits = bits[n:]

    selected_indices = np.flatnonzero(plot_bits).tolist()
    selected_water = sum(
        qubo["plot_weights"][i] for i in selected_indices
    )
    slack_value = sum(
        int(bit) * weight
        for bit, weight in zip(slack_bits, qubo["slack_weights"])
    )

    return {
        "selected_indices": selected_indices,
        "selected_water": int(selected_water),
        "slack_value": int(slack_value),
        "feasible": selected_water <= qubo["water_budget"],
    }


def exact_solve(qubo):
    """Classical exact solver by enumeration."""
    best_bits = None
    best_value = float("inf")
    feasible_count = 0

    for integer in range(2 ** qubo["n_vars"]):
        bits = np.array(
            [(integer >> i) & 1 for i in range(qubo["n_vars"])],
            dtype=int,
        )
        decoded = decode_bits(bits, qubo)
        if decoded["feasible"]:
            feasible_count += 1

        value = qubo_value(bits, qubo)
        if value < best_value:
            best_value = value
            best_bits = bits

    return best_bits, best_value, feasible_count


def qubo_to_ising(qubo):
    """
    Convert:
        f(x) = c + sum a_i x_i + sum b_ij x_i x_j
    using x_i = (1 - Z_i)/2.

    Returns a SparsePauliOp and the constant offset.
    """
    n = qubo["n_vars"]

    z_linear = np.zeros(n, dtype=float)
    zz = {}

    offset = float(qubo["constant"])
    offset += 0.5 * float(np.sum(qubo["linear"]))

    # Linear term a_i*x_i = a_i/2 - a_i/2 Z_i
    z_linear -= 0.5 * qubo["linear"]

    for (i, j), coeff in qubo["quadratic"].items():
        # b*x_i*x_j = b/4 * (1 - Zi - Zj + ZiZj)
        offset += 0.25 * coeff
        z_linear[i] -= 0.25 * coeff
        z_linear[j] -= 0.25 * coeff
        zz[(i, j)] = zz.get((i, j), 0.0) + 0.25 * coeff

    paulis = []

    for i, coeff in enumerate(z_linear):
        if abs(coeff) > 1e-12:
            paulis.append(("Z", [i], float(coeff)))

    for (i, j), coeff in zz.items():
        if abs(coeff) > 1e-12:
            paulis.append(("ZZ", [i, j], float(coeff)))

    if not paulis:
        paulis = [("I", [], 0.0)]

    hamiltonian = SparsePauliOp.from_sparse_list(paulis, num_qubits=n)
    return hamiltonian, offset


def run_qaoa(
    hamiltonian,
    qubo,
    reps: int = 1,
    maxiter: int = 60,
    seed: int = 7,
    shots: int = 4096,
):
    """
    Run QAOA on a local Statevector simulator.

    This intentionally uses the current Qiskit qaoa_ansatz API rather than
    older qiskit.algorithms.QAOA interfaces.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")

    ansatz = qaoa_ansatz(
        cost_operator=hamiltonian,
        reps=reps,
        flatten=True,
    )

    parameters = list(ansatz.parameters)
    rng = np.random.default_rng(seed)
    initial = rng.uniform(-0.25, 0.25, size=len(parameters))

    history = []

    def expectation(theta):
        bound = ansatz.assign_parameters(
            dict(zip(parameters, theta))
        )
        state = Statevector.from_instruction(bound)
        energy = float(
            np.real(state.expectation_value(hamiltonian))
        )
        history.append(energy)
        return energy

    result = minimize(
        expectation,
        initial,
        method="COBYLA",
        options={"maxiter": maxiter, "rhobeg": 0.5},
    )

    optimized_circuit = ansatz.assign_parameters(
        dict(zip(parameters, result.x))
    )
    final_state = Statevector.from_instruction(optimized_circuit)

    counts = final_state.sample_counts(shots=shots)

    # Choose the best feasible sampled bitstring according to the original QUBO.
    candidates = []
    for bitstring, count in counts.items():
        # Qiskit bitstrings are printed with the highest-index qubit first.
        bits = np.array(
            [int(ch) for ch in bitstring[::-1]],
            dtype=int,
        )
        decoded = decode_bits(bits, qubo)
        value = qubo_value(bits, qubo)

        if decoded["feasible"]:
            candidates.append((value, -count, bitstring, bits, count))

    if not candidates:
        # Fall back to the most frequent sampled state.
        bitstring, count = counts.most_common(1)[0]
        bits = np.array([int(ch) for ch in bitstring[::-1]], dtype=int)
    else:
        candidates.sort(key=lambda x: (x[0], x[1]))
        _, _, bitstring, bits, _ = candidates[0]

    return {
        "optimizer_result": result,
        "history": history,
        "circuit": optimized_circuit,
        "statevector": final_state,
        "counts": counts,
        "bits": bits,
        "bitstring": bitstring,
        "objective": qubo_value(bits, qubo),
        "decoded": decode_bits(bits, qubo),
    }


def make_plan_table(df, bits, qubo):
    """Create a user-friendly irrigation plan."""
    n = qubo["n_plots"]
    plan = df.copy()
    plan["irrigate"] = bits[:n].astype(int)
    plan["water_allocated_liters"] = (
        plan["irrigate"] * plan["water_requirement_liters"]
    )
    plan["decision"] = np.where(
        plan["irrigate"] == 1, "IRRIGATE", "SKIP"
    )
    return plan


def save_outputs(
    df,
    qubo,
    exact_bits,
    exact_value,
    qaoa_result,
    output_dir,
):
    """Save CSV, text report and comparison chart."""
    output_dir.mkdir(parents=True, exist_ok=True)

    exact_plan = make_plan_table(df, exact_bits, qubo)
    qaoa_plan = make_plan_table(df, qaoa_result["bits"], qubo)

    comparison = pd.DataFrame(
        [
            {
                "method": "Classical Exact",
                "objective_value": exact_value,
                "water_used_liters": exact_plan[
                    "water_allocated_liters"
                ].sum(),
                "selected_plots": ", ".join(
                    exact_plan.loc[
                        exact_plan["irrigate"] == 1, "plot_id"
                    ]
                ),
            },
            {
                "method": "QAOA",
                "objective_value": qaoa_result["objective"],
                "water_used_liters": qaoa_plan[
                    "water_allocated_liters"
                ].sum(),
                "selected_plots": ", ".join(
                    qaoa_plan.loc[
                        qaoa_plan["irrigate"] == 1, "plot_id"
                    ]
                ),
            },
        ]
    )

    qaoa_plan.to_csv(output_dir / "qaoa_irrigation_plan.csv", index=False)
    exact_plan.to_csv(
        output_dir / "classical_exact_irrigation_plan.csv",
        index=False,
    )
    comparison.to_csv(output_dir / "method_comparison.csv", index=False)

    gap = abs(
        comparison.loc[
            comparison["method"] == "QAOA", "objective_value"
        ].iloc[0]
        - comparison.loc[
            comparison["method"] == "Classical Exact", "objective_value"
        ].iloc[0]
    )

    selected_qaoa = qaoa_plan.loc[
        qaoa_plan["irrigate"] == 1, "plot_id"
    ].tolist()

    selected_exact = exact_plan.loc[
        exact_plan["irrigate"] == 1, "plot_id"
    ].tolist()

    report = f"""
QUANTUM-ASSISTED IRRIGATION OPTIMIZATION
========================================

Water budget: {qubo["water_budget"]} L
Number of plots: {qubo["n_plots"]}
Number of QUBO variables: {qubo["n_vars"]}
QAOA reps: {len(qaoa_result["optimizer_result"].x) // 2}

CLASSICAL EXACT
---------------
Selected plots: {", ".join(selected_exact)}
Water used: {int(exact_plan["water_allocated_liters"].sum())} L
Objective value: {exact_value:.6f}

QAOA
----
Sampled bitstring: {qaoa_result["bitstring"]}
Selected plots: {", ".join(selected_qaoa)}
Water used: {int(qaoa_plan["water_allocated_liters"].sum())} L
Objective value: {qaoa_result["objective"]:.6f}
Optimizer success: {qaoa_result["optimizer_result"].success}
QAOA objective gap vs exact: {gap:.6f}

NOTE
----
QAOA is run on a local statevector simulator. This demonstrates the
algorithmic workflow; it is not a claim of quantum advantage.
"""

    (output_dir / "final_report.txt").write_text(
        report.strip() + "\n",
        encoding="utf-8",
    )

    # Plot 1: water requirement and QAOA decision.
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.18, df["water_requirement_liters"], width=0.36,
           label="Water requirement (L)")
    ax.bar(
        x + 0.18,
        qaoa_plan["water_allocated_liters"],
        width=0.36,
        label="QAOA allocation (L)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(df["plot_id"])
    ax.set_ylabel("Liters")
    ax.set_title("Plot Water Requirement vs QAOA Allocation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "qaoa_irrigation_plan.png", dpi=180)
    plt.close(fig)

    # Plot 2: exact vs QAOA objective.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        comparison["method"],
        comparison["objective_value"],
    )
    ax.set_ylabel("QUBO objective (lower is better)")
    ax.set_title("QAOA vs Classical Exact")
    fig.tight_layout()
    fig.savefig(output_dir / "qaoa_vs_exact.png", dpi=180)
    plt.close(fig)

    return comparison, report


def main():
    parser = argparse.ArgumentParser(
        description="Quantum-assisted irrigation optimization with QUBO + QAOA."
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to irrigation CSV.",
    )
    parser.add_argument(
        "--water-budget",
        type=int,
        default=None,
        help="Available water in liters. If omitted, 60%% of total demand is used.",
    )
    parser.add_argument(
        "--penalty",
        type=float,
        default=2.0,
        help="QUBO constraint penalty.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=1,
        help="QAOA depth.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=60,
        help="Maximum classical optimizer iterations.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=4096,
        help="Statevector sampling shots.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    args = parser.parse_args()

    df = load_dataset(args.data)
    water_budget = (
        args.water_budget
        if args.water_budget is not None
        else choose_default_capacity(df)
    )

    if water_budget <= 0:
        raise ValueError("Water budget must be positive.")

    qubo = build_qubo(
        df,
        water_budget=water_budget,
        penalty=args.penalty,
    )

    exact_bits, exact_value, feasible_count = exact_solve(qubo)
    hamiltonian, offset = qubo_to_ising(qubo)

    qaoa_result = run_qaoa(
        hamiltonian=hamiltonian,
        qubo=qubo,
        reps=args.reps,
        maxiter=args.maxiter,
        seed=args.seed,
        shots=args.shots,
    )

    comparison, report = save_outputs(
        df=df,
        qubo=qubo,
        exact_bits=exact_bits,
        exact_value=exact_value,
        qaoa_result=qaoa_result,
        output_dir=DEFAULT_OUTPUT,
    )

    print("\n" + "=" * 60)
    print("QUANTUM-ASSISTED IRRIGATION OPTIMIZATION")
    print("=" * 60)
    print(f"Water budget: {water_budget} L")
    print(f"QUBO variables: {qubo['n_vars']}")
    print(f"Feasible states checked: {feasible_count}")

    exact_decoded = decode_bits(exact_bits, qubo)
    qaoa_decoded = qaoa_result["decoded"]

    exact_ids = df.iloc[exact_decoded["selected_indices"]]["plot_id"].tolist()
    qaoa_ids = df.iloc[qaoa_decoded["selected_indices"]]["plot_id"].tolist()

    print("\nCLASSICAL EXACT")
    print("----------------")
    print("Selected plots:", ", ".join(exact_ids))
    print("Water used:", exact_decoded["selected_water"], "L")
    print("Objective:", f"{exact_value:.6f}")

    print("\nQAOA")
    print("----")
    print("Selected plots:", ", ".join(qaoa_ids))
    print("Water used:", qaoa_decoded["selected_water"], "L")
    print("Objective:", f"{qaoa_result['objective']:.6f}")
    print("Measured bitstring:", qaoa_result["bitstring"])

    print("\nComparison")
    print("----------")
    print(
        "Objective gap:",
        f"{abs(qaoa_result['objective'] - exact_value):.6f}",
    )

    print("\nFiles saved to:", DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
