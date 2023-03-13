import os
import pprint

import pytest
from train import get_args, train_agent, watch, load_policy
import torch.multiprocessing as mp


# @pytest.mark.skip(reason="runtime too long and unstable result")
def test_mpr(args=get_args()):
    if args.watch:
        watch(args)
        return

    result, masp_policy = train_agent(args)

    # assert result["best_reward"] >= 30.0

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, masp_policy=masp_policy)


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    test_mpr(get_args())
