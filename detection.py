import time
import math
import numpy as np
from collections import defaultdict
from bisect import bisect_left, bisect_right

def gaussian_detection(features, phi=1.96):
    """Assume the residuals have a Gaussian distribution,
       and regard the data that is too far away from mean as an abnormity.

    Args:
        features: a dict whose keys are the names of feature and the values are the corresponding records.
        length: the number of residuals.
        phi: the quantile of the Gaussion distribution.

    Returns:
        a dict whose keys are the names of feature and the values are the corresponding abnormities.
    """
    abnormities = defaultdict(list)
    for feature in features:
        data = features[feature]
        length = len(data) - 1
        res = data[1:] - data[:-1]
        std = np.std(res)
        mean = np.mean(res)
        lb = mean - phi * std
        ub = mean + phi * std
        for i in range(length):
            if res[i] < lb or res[i] > ub:
                abnormities[feature].append(i)
    return abnormities


def str2timestamp(datetimes):
    """Transports a list of strings to a list of timestamp
    """

    def s2t(datetime):
        return time.mktime(time.strptime(datetime, '%Y-%m-%d %H:%M:%S'))

    return list(map(s2t, datetimes))


def nearest_r_abnormity(g_abnormities, r_abnormities):
    """Calculates the time delay between the abnormities found by the algorithm
       and the nearest recorded abnormity after it.

    Args:
        g_abnormities: the abnormities found by the algorithm.
        r_abnormities: the recorded abnormities.

    Returns:
        a list of the time delay between the abnormities found by the algorithm
        and the nearest recorded abnormity after it.
    """
    res = []
    for ts in g_abnormities:
        if ts > r_abnormities[-1]:
            res.append(float("inf"))
        else:
            idx = bisect_left(r_abnormities, ts)
            res.append(r_abnormities[idx] - ts)
    return res


def nearest_g_abnormity(g_abnormities, r_abnormities):
    """Calculates the time delay between the recorded abnormities
       and the nearest the abnormity found by the algorithm before it.

    Args:
        g_abnormities: the abnormities found by the algorithm.
        r_abnormities: the recorded abnormities.

    Returns:
        a list of the time delay between the recorded abnormities
        and the nearest the abnormity found by the algorithm before it.
    """
    res = []
    for ts in r_abnormities:
        if ts < g_abnormities[0]:
            res.append(float("inf"))
        else:
            idx = bisect_right(g_abnormities, ts)
            res.append(ts - g_abnormities[idx - 1])
    return res


def adj_precision(g_abnormities, r_abnormities, weight=lambda x: math.exp(-x / 86400)):
    """Calculate the adjusted precision of the result
       which is weighted by the time delay.

    Args:
        g_abnormities: the abnormities found by the algorithm.
        r_abnormities: the recorded abnormities.

    Returns:
        the adjusted precision
    """
    delay = nearest_g_abnormity(g_abnormities, r_abnormities)
    return sum(map(weight, delay)) / len(delay)


def adj_recall(g_abnormities, r_abnormities, weight=lambda x: math.exp(-x / 86400)):
    """Calculate the adjusted recall of the result
       which is weighted by the time delay.

    Args:
        g_abnormities: the abnormities found by the algorithm.
        r_abnormities: the recorded abnormities.

    Returns:
        the adjusted recall
    """
    delay = nearest_r_abnormity(g_abnormities, r_abnormities)
    return sum(map(weight, delay)) / len(delay)


def adj_f1(g_abnormities, r_abnormities):
    """"Calculate the adjusted F1_score of the result

    Args:
        g_abnormities: a list of datetimes when the algorithm raises an abnormity.
        r_abnormities: a list of datetimes when the there is a recorded abnormity.

    Returns:
        the adjusted F1_score.
    """
    ap = adj_precision(g_abnormities, r_abnormities)
    ar = adj_recall(g_abnormities, r_abnormities)
    return 2 / (1 / ap + 1 / ar)


def weighted_count(abnormities, adj_f1_scores, length):
    """Weights the features by its adjusted F1_score and the
       number of abnormities it raises.

    Args:
        abnormities: a dict whose keys are the names of feature
                     and the values are the abnormities it raises.
        adj_F1_scores: a dict whose keys are the names of feature
                       and the values are corresponding adjusted F1_score.
        length: the number of residuals.

    Returns:
        a list of floats, the number means how many features raise abnormity(weighted) at this moment.
    """
    count = np.zeros(length)
    for feature in abnormities:
        abnormity = abnormities[feature]
        if abnormity:
            weight = adj_f1_scores[feature] / np.log(
                1 + len(abnormity))  # The more abnormities a feature raise, the less important it is
            for dtime in abnormity:
                count[dtime] += weight
    return count


def get_abnormity(count, date, N):
    """Returns the dates in which most abnormities are raised.

    Args:
        count: a list of intergers consisting of the number of abnormities.
        date: a list of date.
        N: the number of chosen abnormal moments.
    """
    dd = defaultdict(lambda: 0)
    for i in np.argsort(-count)[:N]:
        dd[date[i].split(' ')[0]] += 1
    return sorted(sorted([key for key in dd], key=lambda x: -dd[x]))
