from typing import Any, Dict, Tuple

import math
import optuna
from tianshou.trainer.base import BaseTrainer
from tianshou.trainer.utils import test_episode

'''
This class extends from BaseTrainer adding Optuna functionalities.
'''


class BaseOptimizer(BaseTrainer):

    def __init__(self, trial: optuna.Trial, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Two instance variables are added :
        # One to save the trial
        self.trial = trial
        # One to save the number of epochs within a trial (for possible pruning)
        self.eval_idx = 0

    def test_step(self) -> Tuple[Dict[str, Any], bool]:
        """Perform one testing step."""
        assert self.episode_per_test is not None
        assert self.test_collector is not None
        stop_fn_flag = False
        test_result = test_episode(
            self.policy, self.test_collector, self.test_fn, self.epoch,
            self.episode_per_test, self.logger, self.env_step, self.reward_metric
        )
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if self.best_epoch < 0 or self.best_reward < rew:
            self.best_epoch = self.epoch
            self.best_reward = float(rew)
            self.best_reward_std = rew_std
            if self.save_best_fn:
                self.save_best_fn(self.policy)
        if self.verbose:
            print(
                f"Epoch #{self.epoch}: test_reward: {rew:.6f} ± {rew_std:.6f},"
                f" best_reward: {self.best_reward:.6f} ± "
                f"{self.best_reward_std:.6f} in #{self.best_epoch}",
                flush=True
            )
        if not self.is_run:
            test_stat = {
                "test_reward": rew,
                "test_reward_std": rew_std,
                "best_reward": self.best_reward,
                "best_reward_std": self.best_reward_std,
                "best_epoch": self.best_epoch
            }
        else:
            test_stat = {}
        if self.stop_fn and self.stop_fn(self.best_reward):
            stop_fn_flag = True

        # One evaluation is added to the total
        self.eval_idx += 1

        # Average metric is extracted
        avg_spread_factor = test_result['spread_factor_mean']

        # Intermediate report of the trial is printed: metric is the same as the trial
        self.trial.report(avg_spread_factor, self.eval_idx)

        # Prune trial if needed
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return test_stat, stop_fn_flag
