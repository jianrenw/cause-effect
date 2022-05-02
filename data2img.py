import numpy as np
import pandas as pd

def read_causal_pairs(filename):
    """A simplified version of converting the causal pairs, from cdt.io by Diviyan Kalainathan
    Convert a ChaLearn Cause effect pairs challenge format into numpy.ndarray."""
    def convert_row(row):
        a = row["A"].split(" ")
        b = row["B"].split(" ")
        if a[0] == "":
            a.pop(0)
            b.pop(0)
        if a[-1] == "":
            a.pop(-1)
            b.pop(-1)
        a = np.array([float(i) for i in a])
        b = np.array([float(i) for i in b])
        return row['SampleID'], a, b

    data = pd.read_csv(filename)
    conv_data = []
    for _, row in data.iterrows():
        conv_data.append(convert_row(row))
    df = pd.DataFrame(conv_data, columns=['SampleID', 'A', 'B'])
    df = df.set_index("SampleID")
    return df

def read_data(file):
    try:
        df = pd.read_csv(file)
    except pd.io.common.EmptyDataError:
        df = pd.DataFrame()
    return df

def load_data(data_path,label_path=None,size=32, log= True, regression = False):
    """Data loading process
     Args:                data_path: the csv file for x,y causal pairs
                          label_path: the csv file for labels
                          size: the dimension of histogram, for example, 8 indicates a 8x8
                          log: flag for taking the log after the generated histograms
    Return: the generated numpy array of histograms, their labels


     Data should be in the format as the following:
     SampleID, A, B
     pair0, 0.1 0.2 0.5 0.2 ...., 1.2 1.5 1.2 1.1 ...
     For label file, it should also be a csv file that starts with
     SampleID,Target
     pair0,1  pair1,-1, ....
     """
    data_train = []
    target_train = []

    xdata_org = read_causal_pairs(data_path)

    num_rows = len(xdata_org.index)
    for i in range(num_rows):
        x = xdata_org["A"][i]
        y = xdata_org["B"][i]
        H_T = np.histogram2d(x=x, y=y, bins=size)
        H = H_T[0].T
        HT = H / H.max()
        HTX = HT / HT.max()
        if log:
            hislog = np.log10(HTX + 10 ** -8)
        else:
            hislog = HTX
        data_train.append(hislog)

        if label_path != None:
            ydata_org = read_data(label_path).set_index("SampleID")
            target_symbol =  ydata_org["Target"][i]
            if target_symbol == -1:
                target_symbol = 0
            if regression:
                target_symbol =  ydata_org["Details"][i]
            target_train.append(target_symbol)

    xx = np.array(data_train)[:, np.newaxis, :, :]
    if label_path != None:
        return xx,target_train
    else:
        return xx