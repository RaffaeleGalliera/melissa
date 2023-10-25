# Here hyperparameter sets are specified and defined

def l_dgn_params_set(trial, args):
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1, log=True)  # def 0.001
    args.gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])  # def 0.99
    args.buffer_size = trial.suggest_categorical("buffer_size", [int(5e4), int(1e5), int(5e5), int(1e6)])  # def 1e5
    args.hidden_emb = trial.suggest_categorical("hidden_emb", [16, 32, 64, 128, 256, 512])  # def 128
    args.num_heads = trial.suggest_categorical("num_heads", [2, 4, 6])  # def 4
    args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])  # def 32
    args.eps_train_final = trial.suggest_uniform("eps_train_final", 0, 0.2)  # def 0.05
    args.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.5, 0.8)  # def 0.6
    args.update_per_step = trial.suggest_categorical("update_per_step", [0.1, 0.5, 1])  # def 0.1
    args.step_per_collect = trial.suggest_categorical("step_per_collect", [10, 50, 100])  # def 10
    args.target_update_freq = trial.suggest_categorical("target_update_freq", [100, 500, 1000, 5000])  # def 500
    args.aggregator_function = trial.suggest_categorical("aggregator_function",
                                                         ["global_max_pool", "global_add_pool", "global_mean_pool"])
    # def "global_max_pool"


def hl_dgn_params_set(trial, args):
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1, log=True)  # def 0.001
    args.gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])  # def 0.99
    args.buffer_size = trial.suggest_categorical("buffer_size", [int(5e4), int(1e5), int(5e5), int(1e6)])  # def 1e5
    args.hidden_emb = trial.suggest_categorical("hidden_emb", [16, 32, 64, 128, 256, 512])  # def 128
    args.num_heads = trial.suggest_categorical("num_heads", [2, 4, 6])  # def 4
    args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])  # def 32
    args.eps_train_final = trial.suggest_uniform("eps_train_final", 0, 0.2)  # def 0.05
    args.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.5, 0.8)  # def 0.6
    args.update_per_step = trial.suggest_categorical("update_per_step", [0.1, 0.5, 1])  # def 0.1
    args.step_per_collect = trial.suggest_categorical("step_per_collect", [10, 50, 100])  # def 10
    args.target_update_freq = trial.suggest_categorical("target_update_freq", [100, 500, 1000, 5000])  # def 500
    args.aggregator_function = trial.suggest_categorical("aggregator_function",
                                                         ["global_max_pool", "global_add_pool", "global_mean_pool"])
    # def "global_max_pool"


def dgn_r_params_set(trial, args):
    args.lr = trial.suggest_float("learning_rate", 1e-5, 1, log=True)  # def 0.001
    args.gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])  # def 0.99
    args.buffer_size = trial.suggest_categorical("buffer_size", [int(5e4), int(1e5), int(5e5), int(1e6)])  # def 1e5
    args.hidden_emb = trial.suggest_categorical("hidden_emb", [16, 32, 64, 128, 256, 512])  # def 128
    args.num_heads = trial.suggest_categorical("num_heads", [2, 4, 6])  # def 4
    args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256, 512])  # def 32
    args.eps_train_final = trial.suggest_uniform("eps_train_final", 0, 0.2)  # def 0.05
    args.exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.5, 0.8)  # def 0.6
    args.update_per_step = trial.suggest_categorical("update_per_step", [0.1, 0.5, 1])  # def 0.1
    args.step_per_collect = trial.suggest_categorical("step_per_collect", [10, 50, 100])  # def 10
    args.target_update_freq = trial.suggest_categorical("target_update_freq", [100, 500, 1000, 5000])  # def 500
    args.aggregator_function = trial.suggest_categorical("aggregator_function",
                                                         ["global_max_pool", "global_add_pool", "global_mean_pool"])
    # def "global_max_pool"
