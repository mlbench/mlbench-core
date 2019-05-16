import time


def task1_time_to_accuracy_light_goal(metric_name, value, tracker):
    """ Accuracy over Time target for benchmark task 1: Image classification (Light)

    Light target is 70% accuracy

    Args:
        metric_name(str): Name of the metric to test the value for, only "val_Prec@1" is counted
        value (float): Metric value to check
        tracker (`obj`:mlbench_core.utils.tracker.Tracker): Tracker object used for the current run
    Return:
        result (str) or `None` if target is not reached
    """
    if metric_name != "global_val_Prec@1":
        return None

    if value >= 70:
        duration = time.time() - tracker.start_time
        result = "70% Top 1 Validation Accuracy reached in {0:.3f} seconds"\
            .format(duration)
        return result

    return None


def task1_time_to_accuracy_goal(metric_name, value, tracker):
    """ Accuracy over Time target for benchmark task 1: Image classification

    Target is 80% accuracy

    Args:
        metric_name(str): Name of the metric to test the value for, only "val_Prec@1" is counted
        value (float): Metric value to check
        tracker (`obj`:mlbench_core.utils.tracker.Tracker): Tracker object used for the current run
    Return:
        result (str) or `None` if target is not reached
    """
    if metric_name != "global_val_Prec@1":
        return None

    if value >= 80:
        duration = time.time() - tracker.start_time
        result = "80% Top 1 Validation Accuracy reached in {0:.3f} seconds"\
            .format(duration)
        return result

    return None


