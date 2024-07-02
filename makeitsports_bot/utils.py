from makeitsports_bot.data.transforms import inversetransform


def inverse_transform_img(img):
    inv_tensor = inversetransform(img)
    return inv_tensor.permute(1, 2, 0)
