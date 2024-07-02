from makeitsports_bot.constants import BASE_PATH
import os
import numpy as np


def get_data_from_local():
    right_data = [f"{BASE_PATH}/right/{fn}" for fn in os.listdir(f"{BASE_PATH}/right")]
    top_data = [f"{BASE_PATH}/top/{fn}" for fn in os.listdir(f"{BASE_PATH}/top")]

    x = np.array(right_data + top_data)
    y = np.array([ex.replace("right", "left").replace("top", "bottom") for ex in x])

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
