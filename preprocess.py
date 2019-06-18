import os
import datetime
import numpy as np
from openpyxl import load_workbook


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
