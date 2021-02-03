import numpy as np


def get_image_dicts(prop_flat):
    obj_dict, bg_dict = {}, {}
    obj_counter, bg_counter, i = 0, 0, 0

    while len(obj_dict) < len(prop_flat) or len(bg_dict) < len(prop_flat):
        val = prop_flat[i]
        if val == 1 and len(obj_dict) < len(prop_flat):
            obj_dict[obj_counter] = i
            obj_counter += 1
        elif val == 0 and len(bg_dict) < len(prop_flat):
            bg_dict[bg_counter] = i
            bg_counter += 1
        if i == len(prop_flat) - 1:
            i = 0
        else:
            i += 1

    return obj_dict, bg_dict


def build_pos_tensors(x, obj_dict, bg_dict, inds):
    obj_inds = np.array(list(obj_dict.values()))[inds]
    bg_inds = np.array(list(bg_dict.values()))[inds]
    return x[obj_inds], x[bg_inds]