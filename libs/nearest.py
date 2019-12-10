import numpy as np

def nearest(array, value, side=None):
    # based on: http://stackoverflow.com/a/2566508
    #           http://stackoverflow.com/a/3230123
    #           http://stackoverflow.com/a/17119267
    if side not in (None, "gt", "lt"):
        msg = "'side' must be None, 'gt' or 'lt'. Found {}".format(side)
        raise ValueError(msg)

    raveled = np.ravel(array)
    cleaned = raveled[~np.isnan(raveled)]

    if side is None:
        idx = np.argmin(np.abs(cleaned - value))

    else:
        masker, decisor = (
            (np.ma.less_equal, np.argmin)
            if side == "gt" else
            (np.ma.greater_equal, np.argmax))

        diff = cleaned - value
        mask = masker(diff, 0)
        if np.all(mask):
            return None

        masked_diff = np.ma.masked_array(diff, mask)
        idx = decisor(masked_diff)

    return idx