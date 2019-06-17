import os
import datetime
import time
import math
import numpy as np
from openpyxl import load_workbook
from collections import defaultdict
from bisect import bisect_left, bisect_right


def xls2txt(idir, odir, start, end):
    """Extracts columns from xlsx files and fills missing values with 'None',
       and combines the data belongs to the same feature in a txt file.

    Args:
        idir: the address of the folder containing the input files.
        odir: the address of the folder containing the output files.
        start: a list of the time when the records starts, [year, month, day].
        end: a list of the time when the records ends, [year, month, day].
    """

    idirs = os.listdir(idir)
    cur = datetime.datetime(*start, 0, 0)
    endtime = datetime.datetime(*end, 0, 0)
    dtime = datetime.timedelta(minutes=2)
    nextfile = cur - datetime.timedelta(days=1)
    while cur < endtime:
        while nextfile < endtime:
            nextfile += datetime.timedelta(days=1)
            fname = '{}-{}-{}.xlsx'.format(nextfile.year, nextfile.month, nextfile.day)
            if fname in idirs:
                break
        if nextfile >= endtime:
            break
        filename = os.path.join(idir, fname)
        wb = load_workbook(filename)
        sheet = wb.get_sheet_by_name("Sheet1")
        istime = True
        for col in sheet.columns:
            tmp = cur
            if istime:
                istime = False
                timelist = col[2:]
                txtname = os.path.join(odir, 'datetime.txt')
                with open(txtname, 'a+') as f:
                    while cur < timelist[0].value:
                        f.write('{}\n'.format(cur.strftime('%Y-%m-%d %H:%M:%S')))
                        cur += dtime
                    for i in range(len(col[2:])):
                        f.write('{}\n'.format(col[2 + i].value.strftime('%Y-%m-%d %H:%M:%S')))
            elif col[1].value:
                txtname = os.path.join(odir, col[1].value + '.txt')
                with open(txtname, 'a+') as f:
                    while cur < timelist[0].value:
                        f.write('{}\n'.format('None'))
                        cur += dtime
                    for i in range(len(col[2:])):
                        f.write('{}\n'.format(str(col[2 + i].value)))
            cur = tmp
        cur = timelist[-1].value + dtime


def is_number(s):
    """Judges whether a string is a number
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def readdata(filename):
    """Read the data from a file

    Args:
        filename: the address of a txt file

    Returns:
        an np.array contains the data in the file
    """
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if is_number(line):
                data.append(float(line))
            else:
                if len(data) > 0:
                    data.append(data[-1])
                else:
                    data.append(0)
    return np.array(data)


def readdate(filename):
    """Read the datetime from a file

    Args:
        filename: the address of a txt file

    Returns:
        a list contains the datetime in the file
    """
    date = []
    with open(filename, 'r') as f:
        for line in f:
            date.append(line.strip())
    return date


def feature_selection(idir, odir):
    """Deletes the features which are highly correlated with the existing ones.
       And outputs a file consisting of the remaining features.

    Args:
        idir: the address of the folder containing the input files.
        odir: the address of the folder containing the output file.
    """
    ldirs = os.listdir(idir)
    features = {}
    for fname in ldirs:
        if fname == 'datetime.txt': continue
        fpath = os.path.join(idir, fname)
        data = readdata(fpath)
        iscorr = False
        for feature in features:
            cor = np.corrcoef(features[feature], data)[0][1]
            if cor < -0.8 or cor > 0.8:
                iscorr = True
                break
        if iscorr:
            continue
        features[fname] = data
    fname = os.path.join(odir, 'features.txt')
    with open(fname, 'a+') as f:
        for feature in features:
            f.write('{}\n'.format(feature))


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


def main(idir, odir, featuredir, N=50, generatetxt=False, selectfeatures=False):
    if generatetxt:
        start = [2017, 1, 1]
        end = [2017, 9, 1]
        xls2txt(idir, odir, start, end)
    if selectfeatures:
        feature_selection(odir, featuredir)
    date = readdate(os.path.join(odir, 'datetime.txt'))
    length = len(date) - 1
    features = {}
    with open(os.path.join(featuredir, 'features.txt'), 'r') as f:
        for line in f:
            features[line.strip()] = readdata(os.path.join(odir, line.strip()))
    r_abnormities = []
    with open(os.path.join(featuredir, 'abnormities.txt'), 'r') as f:
        for line in f:
            r_abnormities.append(line.strip())
    r_abnormities_ts = str2timestamp(r_abnormities)
    abnormities = gaussian_detection(features, phi=1.96)
    f1_scores = {}
    for feature in abnormities:
        abnormity = abnormities[feature]
        f1_scores[feature] = adj_f1(str2timestamp(date[i] for i in abnormity), r_abnormities_ts)
    count = weighted_count(abnormities, F1_scores, length)
    print(get_abnormity(count, date, N))


if __name__ == '__main__':
    print("begin")
    idir = r'H:\electric\data\xlsx'
    odir = r'H:\electric\data\txt'
    featuredir = r'H:\electric\data'
    main(idir, odir, featuredir, N=50, generatetxt=False, selectfeatures=False)
    print('end')
