from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple, List

import gurobipy as gp
from gurobipy import GRB
from collections import defaultdict


def print_instance_summary(name: str, a_var, m_var, demand, T) -> Tuple[int, int]:
    """Print schedule + stats; return (covered, broadcasts)."""
    sched_by_t = defaultdict(list)
    for (i, t), var in a_var.items():
        if t < T and var.X > 0.5:
            sched_by_t[t].append(i)

    print(f"\n=====  {name}  =====")
    for t in range(T):
        if sched_by_t.get(t):
            print(f"  t={t:02d}: nodes {sorted(sched_by_t[t])} broadcast")

    total_tx = sum(len(v) for v in sched_by_t.values())
    covered   = sum(1 for i in demand if m_var[i, T].X > 0.5)
    ratio     = covered / len(demand) if demand else 0.0
    print(f"Total broadcasts: {total_tx}")
    print(f"Coverage       : {covered}/{len(demand)}  ({ratio:.2%})\n")
    return covered, total_tx


def normalize_ids(d: dict) -> dict:
    d["vertices"]    = [int(v) for v in d["vertices"]]
    d["source_id"]   = int(d["source_id"])
    d["interested"]  = {int(k): int(v) for k, v in d["interested"].items()}
    d["edges_per_t"] = [[(int(u), int(v)) for u, v in Et] for Et in d["edges_per_t"]]
    d["horizon"]     = int(d["horizon"])
    return d


def load_instance(p: Path) -> Dict:
    with open(p, "r") as fp:
        return normalize_ids(json.load(fp))



def build_model(data: Dict, N: int) -> Tuple[
        gp.Model, Dict, Dict, List[int], List[int], int]:
    """
    Build the Phase-1 MILP for interest-aware temporal Steiner broadcast.

    Returns
    -------
    model : gp.Model
        Gurobi model with coverage-maximisation objective.
    m_var : Dict[(int,int), gp.Var]
        Binary “message carried” variables  m_{i,t}.
    a_var : Dict[(int,int), gp.Var]
        Binary “broadcast action” variables a_{i,t}.
    demand : List[int]
        List of interested vertices D_x.
    V : List[int]
        Vertex set.
    T : int
        Horizon.
    """
    V, T = data["vertices"], data["horizon"]
    source = data["source_id"]
    demand = [i for i, flag in data["interested"].items() if flag == 1]
    E_per_t = data["edges_per_t"]          # list of directed (u,v) pairs

    m = gp.Model("Temporal_Steiner")
    m.Params.OutputFlag = 0                # set 1 for verbose Gurobi log

    # ---------------------------------------------------- #
    # Variables
    # ---------------------------------------------------- #
    m_var, a_var = {}, {}
    for i in V:
        for t in range(T + 1):
            m_var[i, t] = m.addVar(
                vtype=GRB.BINARY, name=f"m_{i}_{t}"
            )
            if t < T:
                a_var[i, t] = m.addVar(
                    vtype=GRB.BINARY, name=f"a_{i}_{t}"
                )

    # ---------------------------------------------------- #
    # Initial seed
    # ---------------------------------------------------- #
    m.addConstr(m_var[source, 0] == 1, name="init_source")
    for i in V:
        if i != source:
            m.addConstr(m_var[i, 0] == 0, name=f"init_{i}")

    # ---------------------------------------------------- #
    # Persistence & causal broadcast
    # ---------------------------------------------------- #
    for i in V:
        for t in range(T):
            m.addConstr(
                m_var[i, t + 1] >= m_var[i, t],
                name=f"persist_{i}_{t}"
            )
            m.addConstr(
                a_var[i, t] <= m_var[i, t],
                name=f"causal_{i}_{t}"
            )

    # ---------------------------------------------------- #
    # Reception
    #   m_{j,t+1} ≥ a_{i,t}      ∀(i,j)∈E_t          (OR per arc)
    #   m_{j,t+1} ≤ m_{j,t} + Σ  (i,j)∈E_t a_{i,t}  (upper bound)
    # ---------------------------------------------------- #
    for t, E_t in enumerate(E_per_t):
        # per-arc lower-bound (allows many simultaneous inflows)
        for u, v in E_t:
            m.addConstr(
                m_var[v, t + 1] >= a_var[u, t],
                name=f"recv_arc_{u}_{v}_{t}"
            )

        # single upper-bound inequality for each destination
        incoming = defaultdict(list)
        for u, v in E_t:
            incoming[v].append(u)

        for j in V:
            inflow = gp.quicksum(a_var[u, t] for u in incoming[j])
            m.addConstr(
                m_var[j, t + 1] <= m_var[j, t] + inflow,
                name=f"recv_upper_{j}_{t}"
            )

    # ---------------------------------------------------- #
    # Lifetime window: a_{i,t} ≤ m_{i,t} − m_{i,t−N}
    # ---------------------------------------------------- #
    for i in V:
        for t in range(N, T):
            m.addConstr(
                a_var[i, t] <= m_var[i, t] - m_var[i, t - N],
                name=f"lifetime_{i}_{t}"
            )

    # ---------------------------------------------------- #
    # Phase-1 objective: maximise coverage of interested nodes
    # ---------------------------------------------------- #
    m.setObjective(
        gp.quicksum(m_var[i, T] for i in demand),
        GRB.MAXIMIZE
    )
    m.update()
    return m, m_var, a_var, demand, V, T



# ---------------------------------------------------------------------------
# Solve two-phase MILP for one instance
# ---------------------------------------------------------------------------
def solve_instance(path: Path, N: int, save_sched: bool
                   ) -> Tuple[int, int, int]:
    data = load_instance(path)
    model, m_var, a_var, demand, V, T = build_model(data, N)

    model.optimize()
    C_star = int(model.objVal)

    # Phase-2 (fix coverage, minimise broadcasts)
    model.addConstr(gp.quicksum(m_var[i, T] for i in demand) == C_star)
    model.setObjective(gp.quicksum(a_var[i, t] for i in V for t in range(T)),
                       GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"{path.name}: solver did not reach OPTIMAL")

    # print per-instance schedule + stats
    covered, broadcasts = print_instance_summary(path.name, a_var, m_var,
                                                 demand, T)

    # optional JSON schedule dump
    if save_sched:
        sched = [[(i, t) for i in V if a_var[i, t].X > 0.5] for t in range(T)]
        out = path.with_suffix(".schedule.json")
        with open(out, "w") as fp:
            json.dump(sched, fp, indent=2)

    return covered, len(demand), broadcasts


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------
def mean_std(vals: List[float]) -> Tuple[float, float]:
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return mu, math.sqrt(var)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("MILP solver for temporal broadcast")
    parser.add_argument("--json-file", type=Path, required=True,
                        help="JSON file or directory produced by collect_spacetime.py")
    parser.add_argument("--N", type=int, default=4,
                        help="per-node broadcast budget")
    parser.add_argument("--save-schedule", action="store_true")
    args = parser.parse_args()

    paths = []
    if args.json_file.is_file():
        paths = [args.json_file]
        data = json.loads(open(args.json_file).read())
        print("T recorded =", data["horizon"])
        print("env num_moves stored in JSON name =", args.json_file.stem.split('.')[-1])

    elif args.json_file.is_dir():
        paths = sorted(args.json_file.glob("*.json"))
        if not paths:
            parser.error("Directory contains no *.json files.")
    else:
        parser.error("Provided --json-file is neither a file nor directory.")



    cov_ratios, tx_counts = [], []
    for p in paths:
        covered, demand_sz, broadcasts = solve_instance(
            p, args.N, args.save_schedule
        )
        cov_ratios.append(covered / demand_sz if demand_sz else 0.0)
        tx_counts.append(broadcasts + 1) # +1 for the source broadcast

    if len(paths) > 1:
        mu_cov,  sd_cov  = mean_std(cov_ratios)
        mu_tx,   sd_tx   = mean_std(tx_counts)
        print("\n==========  BATCH SUMMARY  ==========")
        print(f"Instances solved  : {len(paths)}")
        print(f"Coverage ratio    : {mu_cov:.2%} ± {sd_cov:.2%}")
        print(f"Broadcasts (mean) : {mu_tx:.2f} ± {sd_tx:.2f}")
        print("======================================\n")


if __name__ == "__main__":
    main()
