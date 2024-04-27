
from ray.tune.schedulers import AsyncHyperBandScheduler, MedianStoppingRule

def select_sched_alg(sched_alg):
    
    sched = None
    if sched_alg == "AsyncHyperBand":
        sched = AsyncHyperBandScheduler(
            time_attr="training_iteration",
            metric="test_accuracy",
            mode="max",
            max_t=50,
            grace_period=1,
            reduction_factor=3,
            brackets=3,
        )
    elif sched_alg == "MedianStop":
        sched = MedianStoppingRule(
            time_attr="time_total_s",
            metric="test_accuracy",
            mode="max",
            grace_period=1,
            min_samples_required=3,
        )
    else:
        print("Unknown Option. Select MedianStop or AsyncHyperBand")
    return sched