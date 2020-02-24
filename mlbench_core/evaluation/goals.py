def _add_detailed_times(result, tracker):
    compute_time = tracker.get_total_compute_time()

    if compute_time:
        result += ", Compute: {} seconds".format(compute_time)

    communication_time = tracker.get_total_communication_time()

    if communication_time:
        result += ", Communication: {} seconds".format(communication_time)

    return result


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
    if metric_name != "val_global_Prec@1":
        return None

    if value >= 70:
        duration = tracker.get_total_train_time()

        result = "70% Top 1 Validation Accuracy reached in {0:.3f} seconds"\
            .format(duration)

        result = _add_detailed_times(result, tracker)

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
    if metric_name != "val_global_Prec@1":
        return None

    if value >= 80:
        duration = tracker.get_total_train_time()
        result = "80% Top 1 Validation Accuracy reached in {0:.3f} seconds"\
            .format(duration)

        result = _add_detailed_times(result, tracker)

        return result

    return None


def task3_time_to_preplexity_goal(metric_name, value, tracker):
    """Time to perplexity goal for benchmark task 3: Language Modelling

    Target is a perplexity of 50

    Args:
        metric_name(str): Name of the metric to test the value for, only "val_Prec@1" is counted
        value (float): Metric value to check
        tracker (`obj`:mlbench_core.utils.tracker.Tracker): Tracker object used for the current run
    Return:
        result (str) or `None` if target is not reached
    """

    if metric_name != "val_global_Perplexity":
        return None

    if value <= 50:
        duration = tracker.get_total_train_time()
        result = "Validation perplexity of 50 reached in {0:.3f} seconds"\
            .format(duration)

        result = _add_detailed_times(result, tracker)

        return result

    return None


def task3_time_to_preplexity_light_goal(metric_name, value, tracker):
    """Time to perplexity goal for benchmark task 3: Language Modelling

    Target is a perplexity of 50

    Args:
        metric_name(str): Name of the metric to test the value for, only "val_Prec@1" is counted
        value (float): Metric value to check
        tracker (`obj`:mlbench_core.utils.tracker.Tracker): Tracker object used for the current run
    Return:
        result (str) or `None` if target is not reached
    """

    if metric_name != "val_global_Perplexity":
        return None

    if value <= 100:
        duration = tracker.get_total_train_time()
        result = "Validation perplexity of 50 reached in {0:.3f} seconds"\
            .format(duration)

        result = _add_detailed_times(result, tracker)

        return result

    return None
