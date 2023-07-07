import os.path
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from train import get_args, train_agent
from plotly.graph_objects import Figure


def objective(trial):
    # Get args, where hyperparams are defined and let optuna change them
    args = get_args()

    # Here hyperparameters are suggested by the framework
    args.lr = trial.suggest_float("lr", 1e-5, 1, log=True)
    args.gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    args.batch_size = trial.suggest_int("batch_size", 32, 128)

    # Debug Line: print("Iperparameters (lr,gamma,batch_size): " + str(args.lr) + ", " + str(args.gamma) + ",
    # " + str(args.batch_size))

    # Final result of the trial is saved and returned
    result, _ = train_agent(args, opt_trial=trial)
    return result["best_reward"]


def hyperparams_opt():
    args = get_args()
    # Set number of torch threads
    torch.set_num_threads(1)

    # Here the sampler is defined
    sampler = TPESampler(n_startup_trials=args.n_startup_trials)

    # Here the pruner is defined
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials, n_warmup_steps=args.n_warmup_steps
    )

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
    study = optuna.create_study(study_name=study_name, storage=storage_name, sampler=sampler, pruner=pruner,
                                direction="maximize", load_if_exists=True)

    # Info about study are printed, dataframe is empty if the study is new
    info_study = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    print(info_study)

    try:
        # Start optimization with arguments provided
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

    # Plot and how optimization history and param importances
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig1.show()
    fig2.show()
