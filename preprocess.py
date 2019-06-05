import os
import datetime
import numpy as np
from openpyxl import load_workbook
from collections import defaultdict


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


def gaussian_detection(features, length, phi=1.96):
    """Assume the residuals have a Gaussian distribution,
       and regard the data that is too far away from mean as an abnormal point.

    Args:
        features: a dict whose keys are the names of feature and the values are the corresponding records.
        length: the number of residuals.
        phi: the quantile of the Gaussion distribution.

    Returns:
        a list of intergers, the number means how many features raise abnormity at this moment.
    """
    count = np.zeros(length)
    for feature in features:
        data = features[feature]
        res = data[1:] - data[:-1]
        std = np.std(res)
        mean = np.mean(res)
        lb = mean - phi * std
        ub = mean + phi * std
        for i in range(length):
            if res[i] < lb or res[i] > ub:
                count[i] += 1
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
    length = len(date)-1
    features = {}
    with open(os.path.join(featuredir, 'features.txt'), 'r') as f:
        for line in f:
            features[line] = readdata(os.path.join(odir, line.strip()))
    count = gaussian_detection(features, length, phi=1.96)
    print(get_abnormity(count, date, N))

if __name__ == '__main__':
    print("begin")
    idir = r'H:\electric\data\xlsx'
    odir = r'H:\electric\data\txt'
    featuredir = r'H:\electric\data'
    main(idir, odir, featuredir, N=50, generatetxt=False, selectfeatures=False)
    print('end')