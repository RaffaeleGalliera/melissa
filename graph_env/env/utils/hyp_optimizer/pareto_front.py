from typing import Sequence, List, Optional

import numpy as np
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna.visualization._plotly_imports import go
from optuna.multi_objective.visualization._pareto_front import _make_hovertext

'''
This collection allows a multi-objective Pareto front plot while keeping the optuna report functionality 
(necessary for both pruning and real-time web dashboard). 
These methods are taken and modified from the optuna.visualization framework to fit this specific problem.
'''


def get_pareto_front_trials(
        trials: Sequence[FrozenTrial], directions: Sequence[StudyDirection]) -> List[FrozenTrial]:
    trials_state = np.zeros(len(trials))

    n_trials = len(trials)
    if n_trials == 0:
        return []

    trials.sort(
        key=lambda trial: (
            normalize_value(trial.user_attrs['cov'], directions[0]),
            normalize_value(trial.user_attrs['msg'], directions[1]),
        ),
    )

    last_nondominated_trial = trials[0]
    pareto_front = [last_nondominated_trial]
    trials_state[0] = 1

    for i in range(1, n_trials):
        trial = trials[i]
        if dominates(last_nondominated_trial, trial, directions):
            continue
        pareto_front.append(trial)
        trials_state[i] = 1
        last_nondominated_trial = trial

    pareto_front.sort(key=lambda trial: trial.number)
    return pareto_front, trials_state


def plot_pareto_front(completed_trials, trials_state):
    layout = go.Layout(
        title="Pareto-front Plot",
        xaxis_title="Coverage",
        yaxis_title="Messages",
    )

    colors = [None] * len(completed_trials)
    for i in range(0, len(completed_trials)):
        if trials_state[i] == 1:
            colors[i] = 'red'
        elif trials_state[i] == 2:
            colors[i] = 'green'
        else:
            colors[i] = 'blue'

    # Create a list of scatter traces with individual colors
    traces = []
    for i in range(len(completed_trials)):
        traces.append(
            go.Scatter(
                x=[trial.user_attrs['cov'] for trial in completed_trials],
                y=[trial.user_attrs['msg'] for trial in completed_trials],
                text=[_make_hovertext(trial) for trial in completed_trials],
                mode="markers",
                marker=dict(size=10, color=colors[i]),
                showlegend=False,
            )
        )

    return go.Figure(data=traces, layout=layout)


def dominates(
        trial0: FrozenTrial, trial1: FrozenTrial, directions: Sequence[StudyDirection]
) -> bool:
    values0 = trial0.user_attrs
    values1 = trial1.user_attrs

    assert values0 is not None
    assert values1 is not None

    if len(values0) != len(values1):
        raise ValueError("Trials with different numbers of objectives cannot be compared.")

    if len(values0) != len(directions):
        raise ValueError(
            "The number of the values and the number of the objectives are mismatched."
        )

    if trial0.state != TrialState.COMPLETE:
        return False

    if trial1.state != TrialState.COMPLETE:
        return True

    normalized_values0 = [normalize_value(v, d) for v, d in zip(list(values0.values()), directions)]
    normalized_values1 = [normalize_value(v, d) for v, d in zip(list(values1.values()), directions)]

    if normalized_values0 == normalized_values1:
        return False

    return all(v0 <= v1 for v0, v1 in zip(normalized_values0, normalized_values1))


def normalize_value(value: Optional[float], direction: StudyDirection) -> float:
    if value is None:
        value = float("inf")

    if direction is StudyDirection.MAXIMIZE:
        value = -value

    return value
