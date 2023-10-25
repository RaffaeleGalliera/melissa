import os.path

import torch
import optuna
import logging

from graph_env.env.utils.hyp_optimizer.pareto_front import get_pareto_front_trials, plot_pareto_front
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, NopPruner, BasePruner
from optuna.samplers import TPESampler, RandomSampler, BaseSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState
from optuna.visualization import plot_optimization_history, plot_param_importances
from tianshou.env import DummyVectorEnv
from graph_env.env.utils.collectors.collector import MultiAgentCollector
from plotly.graph_objects import Figure

logging.getLogger().setLevel(logging.INFO)


# This class allows every function in this scope to access algorithm-specific information
class TargetAlgorithm:
    def __init__(self):
        self.get_args = None
        self.train_agent = None
        self.get_env = None
        self.params_set = None

    def set(self, get_args, train_agent, get_env, params_set):
        self.get_args = get_args
        self.train_agent = train_agent
        self.get_env = get_env
        self.params_set = params_set


target_algorithm = TargetAlgorithm()


def objective(trial: optuna.Trial):
    # Get args, where hyperparams are defined and let optuna change them
    args = target_algorithm.get_args()

    # It is possible to create different hyperparams set for different algorithms or for the same one
    target_algorithm.params_set(trial, args)

    # Agents are trained
    train_result, masp_policy = target_algorithm.train_agent(args, opt_trial=trial)

    # Agents are tested
    test_result = test_agent(args, masp_policy)

    # Average metric is extracted
    avg_spread_factor = test_result['spread_factor_mean']

    # Average coverage and messages are saved for Pareto front plot
    trial.set_user_attr("cov", test_result["coverage"])
    trial.set_user_attr("msg", test_result["msg"])

    # The metric result is returned
    return avg_spread_factor


def test_agent(args, masp_policy):
    # Testing environment are extracted
    env = DummyVectorEnv([lambda: target_algorithm.get_env(number_of_agents=args.n_agents,
                                          is_testing=True,
                                          render_mode=None,
                                          dynamic_graph=args.dynamic_graph)])

    # Policy is evaluated
    masp_policy.policy.eval()
    masp_policy.policy.set_eps(args.eps_test)
    collector = MultiAgentCollector(masp_policy, env, exploration_noise=True,
                                    number_of_agents=args.n_agents)

    # Test results are collected
    result = collector.collect(n_episode=args.test_num)
    return result


def create_sampler(args) -> BaseSampler:
    if args.sampler_method == "random":
        sampler = RandomSampler()
    elif args.sampler_method == "tpe":
        sampler = TPESampler(n_startup_trials=args.n_trials // 5, multivariate=True)
    elif args.sampler_method == "skopt":
        from optuna.integration.skopt import SkoptSampler
        sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
    else:
        raise ValueError(f"Unknown sampler: {args.sampler_method}")
    return sampler


def create_pruner(args) -> BasePruner:
    if args.pruner_method == "halving":
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif args.pruner_method == "median":
        pruner = MedianPruner(n_startup_trials=args.n_trials // 5, n_warmup_steps=args.epoch // 3)
    elif args.pruner_method == "none":
        # Do not prune
        pruner = NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {args.pruner_method}")
    return pruner


def hyperparams_opt(get_args, train_agent, get_env, params_set):

    # Algorithm-specific arguments, train_function, environment and set of hyperparameters are set
    target_algorithm.set(get_args, train_agent, get_env, params_set)
    args = target_algorithm.get_args()

    # Set number of torch threads
    torch.set_num_threads(1)

    # Here the sampler is defined
    sampler = create_sampler(args)

    # Here the pruner is defined
    pruner = create_pruner(args)

    # Study name is initialized
    study_name = args.study_name

    # If the study needs to be saved, a database is created
    if args.save_study:
        path_databases = "hyp_studies/databases/"
        if not os.path.exists(path_databases):
            os.makedirs(path_databases)
        storage_name = "sqlite:///{}.db".format(path_databases + study_name)
    else:
        storage_name = None

    # A new study is created (if didn't exist) with name, storage, sampler and pruner specified
    study = optuna.create_study(study_name=study_name,
                                storage=storage_name,
                                sampler=sampler,
                                pruner=pruner,
                                direction="maximize",
                                load_if_exists=True)

    if args.save_study:
        # Prints command to launch optuna-dashboard
        dashboard_command = "optuna-dashboard " + storage_name
        print(dashboard_command)

    # Info about study are printed, dataframe is empty if the study is new
    info_study = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    logging.debug(info_study)

    try:
        # Start optimization with the arguments provided
        study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs, timeout=args.timeout)
    except KeyboardInterrupt:
        pass

    # Completed trials are extracted
    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    # Best trial according to metric is extracted
    best_metric_trial = study.best_trial
    # Trials on the Pareto front are extracted, state of all trials is calculated for color plot
    best_trials, trials_state = get_pareto_front_trials(completed_trials, [StudyDirection(2), StudyDirection(1)])
    # Best trial according to the metric is marked to be colored with a different color in the pareto plot
    for i in range(0, len(completed_trials)):
        if completed_trials[i].number == best_metric_trial.number:
            trials_state[i] = 2

    # General results are printed
    print("Total number of trials: ", len(study.trials))
    print("Number of completed trials: ", len(completed_trials))

    # Best trial information
    print("Best trial according to metric:")
    print(f"   Number: {best_metric_trial.number}")
    print(f"  Value: {best_metric_trial.value}")

    print("  Params: ")
    for key, value in best_metric_trial.params.items():
        print(f"    {key}: {value}")

    print("  Metrics:")
    for key, value in best_metric_trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Pareto Front trials information
    print(f"\nNumber of trials on the Pareto front: {len(best_trials)}\n")
    print("--- Best Trials ---")
    for trial in best_trials:
        print("Trial " + str(trial.number) + ": [cov=" + str(trial.user_attrs['cov']) + ", msg=" + str(trial.user_attrs['msg']) + "]")
        print("Hyperparameters set = " + str(trial.params))

    trial_with_highest_coverage = max(best_trials, key=lambda t: t.user_attrs['cov'])
    print(f"Trial with highest coverage: ")
    print(f"\tnumber: {trial_with_highest_coverage.number}")
    print(f"\tparams: {trial_with_highest_coverage.params}")
    print(f"\tvalues: {trial_with_highest_coverage.user_attrs}")

    trial_with_lowest_messages = min(best_trials, key=lambda t: t.user_attrs['msg'])
    print(f"Trial with lowest messages: ")
    print(f"\tnumber: {trial_with_lowest_messages.number}")
    print(f"\tparams: {trial_with_lowest_messages.params}")
    print(f"\tvalues: {trial_with_lowest_messages.user_attrs}")

    # Write and save trial report on a csv
    path_results = "hyp_studies/results/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Write report
    study.trials_dataframe().to_csv(path_results + study_name + "_result.csv")

    # For study persistence in memory
    # with open("study.pkl", "wb+") as f:
    # pkl.dump(study, f)

    # Plot optimization history, hyperparameter importance and pareto front
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_pareto_front(completed_trials, trials_state)

    fig1.show()
    fig2.show()
    fig3.show()


