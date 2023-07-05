import pprint
from train_nvdn import get_args, train_agent, watch, load_policy


def test_mpr(args=get_args()):
    if args.watch:
        watch(args)
        return

    result, masp_policy = train_agent(args)

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        watch(args, masp_policy=masp_policy)


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)
    test_mpr(get_args())
