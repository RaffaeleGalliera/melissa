from tianshou.utils import WandbLogger


class CustomLogger(WandbLogger):
    def log_train_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during training.

        :param collect_result: a dict containing information of data collected in
            training stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        if collect_result["n/graphs"] > 0:
            log_data = {
                "train/episode": collect_result["n/ep"],
                "train/reward": collect_result["rew"],
                "train/length": collect_result["len"],
                "train/graph": collect_result["n/graphs"],
                "train/coverage": collect_result["coverage"],
                "train/coverage_std": collect_result["coverage_std"],
                "train/messages": collect_result["msg"],
                "train/messages_std": collect_result["msg_std"]
            }
            self.write("train/env_step", step, log_data)
        elif collect_result["n/ep"] > 0:
            if step - self.last_log_train_step >= self.train_interval:
                log_data = {
                    "train/episode": collect_result["n/ep"],
                    "train/reward": collect_result["rew"],
                    "train/length": collect_result["len"],
                }
                self.write("train/env_step", step, log_data)
                self.last_log_train_step = step

    def log_test_data(self, collect_result: dict, step: int) -> None:
        """Use writer to log statistics generated during evaluating.

        :param collect_result: a dict containing information of data collected in
            evaluating stage, i.e., returns of collector.collect().
        :param int step: stands for the timestep the collect_result being logged.
        """
        assert collect_result["n/ep"] > 0
        if collect_result["n/graphs"] > 0:
            log_data = {
                "test/episode": collect_result["n/ep"],
                "test/reward": collect_result["rew"],
                "test/length": collect_result["len"],
                "test/coverage": collect_result["coverage"],
                "test/coverage_std": collect_result["coverage_std"],
                "test/messages": collect_result["msg"],
                "test/messages_std": collect_result["msg_std"]
            }
            self.write("train/env_step", step, log_data)
        elif step - self.last_log_test_step >= self.test_interval:
            log_data = {
                "test/env_step": step,
                "test/reward": collect_result["rew"],
                "test/length": collect_result["len"],
                "test/reward_std": collect_result["rew_std"],
                "test/length_std": collect_result["len_std"],
            }
            self.write("test/env_step", step, log_data)
            self.last_log_test_step = step
