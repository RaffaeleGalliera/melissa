import os.path
import math
import torch
import optuna
import logging

from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, NopPruner, BasePruner
from optuna.samplers import TPESampler, RandomSampler, BaseSampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from tianshou.env import DummyVectorEnv
from graph_env.env.utils.collectors.collector import MultiAgentCollector
from train_dqn import get_args, train_agent, get_env
from .constants import NUMBER_OF_AGENTS

logging.getLogger().setLevel(logging.INFO)


def dqn_params_set(trial, args):
    # Here hyperparameters are suggested to the framework
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1, log=True)  # def 0.001
    args.gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])  # def 0.99
    args.buffer_size = trial.suggest_categorical("buffer_size", [int(5e4), int(1e5), int(5e5), int(1e6)])  # def 1e5
    args.hidden_emb = trial.suggest_categorical("hidden_emb", [16, 32, 64, 128, 256, 512])  # def 128
    args.num_heads = trial.suggest_categorical("num_heads", [2, 4, 6])  # def 4
    args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])  # def 32
    args.eps_train_final = trial.suggest_uniform("eps_train_final", 0, 0.2)  # def 0.05
    args.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.5, 0.8)  # def 0.6
    args.update_per_step = trial.suggest_categorical("update_per_step", [0.1, 0.5, 1])  # def 0.1
    args.step_per_collect = trial.suggest_categorical("step_per_collect", [5, 10, 50, 100])  # def 10
    args.target_update_freq = trial.suggest_categorical("target_update_freq", [100, 500, 1000, 5000])  # def 500
    args.aggregator_function = trial.suggest_categorical("aggregator_function",
                                                         ["global_max_pool", "global_add_pool", "global_mean_pool"]) # def "global_max_pool"


def objective(trial):
    # Get args, where hyperparams are defined and let optuna change them
    args = get_args()

    # It is possible to create different hyperparams set for different algorithms or for the same one
    if args.learning_algorithm == "dqn":
        dqn_params_set(trial, args)

    # Agents are trained
    train_result, masp_policy = train_agent(args, opt_trial=trial)

    # Agents are tested
    test_result = test_agent(args, masp_policy)

    # Average metric is extracted
    avg_spread_factor = test_result['spread_factor_mean']

    # This metric force the reward to be in the [0,1] range and multiplies it by the coverage
    return avg_spread_factor


def test_agent(args, masp_policy):
    # Testing environment are extracted
    env = DummyVectorEnv([lambda: get_env(is_testing=True, render_mode=None, dynamic_graph=args.dynamic_graph)])

    # Policy is evaluated
    masp_policy.policy.eval()
    masp_policy.policy.set_eps(args.eps_test)
    collector = MultiAgentCollector(masp_policy, env, exploration_noise=True, number_of_agents=NUMBER_OF_AGENTS)

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


def hyperparams_opt():
    args = get_args()

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

    # Print final results after all the trials
    trial = study.best_trial

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write and save trial report on a csv
    path_results = "hyp_studies/results/"
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Write report
    study.trials_dataframe().to_csv(path_results + study_name + "_result.csv")

    # For study persistence in memory
    # with open("study.pkl", "wb+") as f:
    # pkl.dump(study, f)

    # Plot optimization history and param importance
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig1.show()
    fig2.show()

