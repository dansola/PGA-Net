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