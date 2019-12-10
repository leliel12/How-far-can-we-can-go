import numpy as np
from sklearn.metrics.ranking import _binary_clf_curve


def prec_star(y_true, probas_pred, ss, rs, pos_label=None,
                           sample_weight=None):
    fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred,
                                             pos_label=pos_label,
                                             sample_weight=sample_weight)
    
    fps = fps * rs / float(ss)
    
    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)
    return np.r_[precision[sl], 1] 



    