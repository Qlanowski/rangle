import numpy as np
import math

def get_preds(hms, img_shape):
    preds = np.zeros((hms.shape[-1], 2))
    sh = img_shape[0] / hms.shape[0]
    sw = img_shape[1] / hms.shape[1]

    for j in range(hms.shape[-1]):
        hm = hms[:, :, j]
        idx = hm.argmax()
        max_y, max_x = np.unravel_index(idx, hm.shape)
        hms[max_y, max_x, j] = float('-inf')

        idx = hm.argmax()
        sec_y, sec_x = np.unravel_index(idx, hm.shape)

        diff = math.sqrt(((max_y-sec_y)**2)+((max_x-sec_x)**2))

        dy = (sec_y - max_y)/diff
        dx = (sec_x - max_x)/diff

        x = max_x + 0.25 * dx
        y = max_y + 0.25 * dy
        preds[j, 0] = x * sw
        preds[j, 1] = y * sh

    return preds