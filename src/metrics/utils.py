import numpy as np


def display_metric_list(my_list, metric_name):
    class0, class1, class2 = [], [], []
    for iou in my_list:
        class0.append(iou[1].detach().cpu().numpy()[0])
        class1.append(iou[1].detach().cpu().numpy()[1])
        class2.append(iou[1].detach().cpu().numpy()[2])
    print(f"Mean water {metric_name}: {np.mean(class0)}")
    print(f"Mean anchor {metric_name}: {np.mean(class1)}")
    print(f"Mean frazil {metric_name}: {np.mean(class2)}")
    print(f"Mean {metric_name}: {np.mean(class0 + class1 + class2)}")


def early_stopping(epoch_metric_list, tolerance):
    max_metric = epoch_metric_list[0]
    lower_count = 0
    stopping_epoch = 0
    for i, x in enumerate(epoch_metric_list):
        if lower_count >= tolerance:
            stopping_epoch = i - tolerance
            break
        if x < max_metric:
            lower_count += 1
        else:
            max_metric = x
            lower_count = 0
    if stopping_epoch:
        return stopping_epoch
    else:
        return len(epoch_metric_list) - lower_count + 1


def early_stopping_loss(epoch_metric_list, tolerance):
    min_metric = epoch_metric_list[0]
    higher_count = 0
    stopping_epoch = 0
    for i, x in enumerate(epoch_metric_list):
        if higher_count >= tolerance:
            stopping_epoch = i - tolerance
            break
        if x > min_metric:
            higher_count += 1
        else:
            min_metric = x
            higher_count = 0
    if stopping_epoch:
        return stopping_epoch
    else:
        return len(epoch_metric_list) - higher_count + 1


def get_mean_window(epoch_metric_list, stopping_epochs, avg_half_window):
    stop = early_stopping(epoch_metric_list, stopping_epochs)
    epoch_ind = stop - 1
    mean_metric = np.mean(epoch_metric_list[epoch_ind - avg_half_window:epoch_ind + avg_half_window])
    return mean_metric

