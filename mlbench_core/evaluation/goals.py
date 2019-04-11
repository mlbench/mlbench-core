import time


def task1_time_to_accuracy_light_goal(metric_name, value, tracker):
    if metric_name != "val_Prec@1":
        return None

    if value >= 0.8:
        duration = time.time() - tracker.start_time
        result = "80% Top 1 Validation Accuracy reached in {0:.3f} seconds"\
            .format(duration)
        return ("TaskResult", result)

    return None


def task1_time_to_accuracy_goal(metric_name, value, tracker):
    if metric_name != "val_Prec@1":
        return None

    if value >= 0.9:
        duration = time.time() - tracker.start_time
        result = "90% Top 1 Validation Accuracy reached in {0:.3f} seconds"\
            .format(duration)
        return ("TaskResult", result)

    return None


