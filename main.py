import os
from preprocess import xls2txt, readdata, readdate
from feature_selection import feature_selection
from detection import str2timestamp, gaussian_detection, adj_f1, weighted_count, get_abnormity

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
    count = weighted_count(abnormities, f1_scores, length)
    print(get_abnormity(count, date, N))


if __name__ == '__main__':
    print("begin")
    idir = r'H:\electric\data\xlsx'
    odir = r'H:\electric\data\txt'
    featuredir = r'H:\electric\data'
    main(idir, odir, featuredir, N=50, generatetxt=False, selectfeatures=False)
    print('end')
