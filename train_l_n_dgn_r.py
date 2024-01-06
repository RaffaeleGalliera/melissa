import os
import pprint

from l_n_dgn_r import get_args, train_agent, watch, get_env
from graph_env.env.utils.optimizer import hyperparams_opt
from graph_env.env.utils.hyp_optimizer.params_set import dgn_r_params_set


# @pytest.mark.skip(reason="runtime too long and unstable result")
def test_mpr(args=get_args()):
    if args.watch:
        watch(args)
        return

    if args.optimize:
        hyperparams_opt(get_args, train_agent, get_env, dgn_r_params_set)
        return

    result, masp_policy = train_agent(args)

    if __name__ == '__main__':
        pprint.pprint(result)


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    test_mpr(get_args())
