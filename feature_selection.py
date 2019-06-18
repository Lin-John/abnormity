import os
import numpy as np
from preprocess import readdata


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
