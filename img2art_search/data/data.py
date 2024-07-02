from img2art_search.constants import BASE_PATH
import os
import numpy as np


def get_data_from_local():
    left_or_top_data = [f"{BASE_PATH}/splits/left_or_top/{fn}" for fn in os.listdir(f"{BASE_PATH}/splits/left_or_top")]
    
    x = np.array(left_or_top_data)
    y = np.array([ex.replace("left_or_top", "right_or_bottom") for ex in x])

    data = np.array([x, y])
    return data


def split_train_val_test(data, test_size, val_size):
    train_size = 1 - test_size - val_size
    SPLIT = int(data.shape[1] * train_size)
    TEST_SPLIT = SPLIT + int(data.shape[1] * test_size)
    train = data[:, :SPLIT]
    validation = data[:, SPLIT:TEST_SPLIT]
    test = data[:, TEST_SPLIT:]
    return train, validation, test
