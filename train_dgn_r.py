import os
import pprint

from dgn_r import get_args, train_agent, watch
from graph_env.env.utils.optimizer import hyperparams_opt


# @pytest.mark.skip(reason="runtime too long and unstable result")
def test_mpr(args=get_args()):
    if args.watch:
        watch(args)
        return

    if args.optimize:
        hyperparams_opt()
        return

    result, masp_policy = train_agent(args)

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, masp_policy=masp_policy)


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    test_mpr(get_args())
