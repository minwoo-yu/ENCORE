import numpy as np
import random


def patch_position(patch_size):
    ih, iw = 512, 512
    iy = random.randrange(0, ih - patch_size + 1)
    ix = random.randrange(0, iw - patch_size + 1)

    patch_coord = np.array(
        [
            [
                (iy + patch_size // 2 - ih // 2) / (ih // 2),
                (ix + patch_size // 2 - iw // 2) / (iw // 2),
            ]
        ]
    )
    return patch_coord


def augment(sino, hflip=True, vflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = vflip and random.random() < 0.5
    rot90 = rot * random.randint(0, 3)
    view, det = sino.shape
    if hflip:
        zero = np.expand_dims(sino[0, :], axis=0)
        sino = np.concatenate((zero, sino[:0:-1, ::-1]), axis=0)
    if vflip:
        zero = np.expand_dims(sino[0, :], axis=0)
        sino = np.concatenate((np.flip(zero, -1), sino[:0:-1, ::-1]), axis=0)
        sino = np.roll(sino, -view // 2, axis=0)
    if rot90:
        flip_len = (rot90 * view // 4) % view
        sino = np.roll(sino, -flip_len, axis=0)
    return sino.copy()
