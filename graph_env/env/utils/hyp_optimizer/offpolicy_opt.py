from typing import Any, Callable, Dict, Optional, Union
import numpy as np
import optuna

from tianshou.data import Collector
from tianshou.policy import BasePolicy
from tianshou.trainer.base import BaseTrainer
from tianshou.utils import BaseLogger, LazyLogger

from .base_opt import BaseOptimizer

'''
Since python doesn't support multiple hereditary, the OffPolicyTrainer class needs to be copied to 
work as wrapper for the BaseOptimizer class.
'''


class OffpolicyOptimizer(BaseOptimizer):
    #__doc__ = BaseTrainer.gen_doc("offpolicy") + "\n".join(__doc__.split("\n")[1:])

    def __init__(
            self,
            policy: BasePolicy,
            train_collector: Collector,
            test_collector: Optional[Collector],
            max_epoch: int,
            step_per_epoch: int,
            step_per_collect: int,
            episode_per_test: int,
            batch_size: int,
            update_per_step: Union[int, float] = 1,
            train_fn: Optional[Callable[[int, int], None]] = None,
            test_fn: Optional[Callable[[int, Optional[int]], None]] = None,
            stop_fn: Optional[Callable[[float], bool]] = None,
            save_best_fn: Optional[Callable[[BasePolicy], None]] = None,
            save_checkpoint_fn: Optional[Callable[[int, int, int], str]] = None,
            resume_from_log: bool = False,
            reward_metric: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            verbose: bool = True,
            show_progress: bool = True,
            test_in_train: bool = True,
            trial: optuna.Trial = None,
            **kwargs: Any,
    ):
        super().__init__(
            learning_type="offpolicy",
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=max_epoch,
            step_per_epoch=step_per_epoch,
            step_per_collect=step_per_collect,
            episode_per_test=episode_per_test,
            batch_size=batch_size,
            update_per_step=update_per_step,
            train_fn=train_fn,
            test_fn=test_fn,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            save_checkpoint_fn=save_checkpoint_fn,
            resume_from_log=resume_from_log,
            reward_metric=reward_metric,
            verbose=verbose,
            show_progress=show_progress,
            test_in_train=test_in_train,
            trial=trial,
            **kwargs,
        )

    def policy_update_fn(self, data: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Perform off-policy updates."""
        assert self.train_collector is not None
        for _ in range(round(self.update_per_step * result["n/st"])):
            self.gradient_step += 1
            losses = self.policy.update(self.batch_size, self.train_collector.buffer)
            self.log_update_data(data, losses)


def offpolicy_optimizer(*args, **kwargs) -> Dict[str, Union[float, str]]:  # type: ignore
    # Wrapper for OffPolicyOptimizer run method.
    return OffpolicyOptimizer(*args, **kwargs).run()


offpolicy_optimizer_iter = OffpolicyOptimizer
