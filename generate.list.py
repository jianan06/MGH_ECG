# generating data
# Sep 19th 2020
import os
import pandas as pd
from tqdm import tqdm


# find labels file
filename = r"Z:\Datasets_ConvertedData\sleeplab\grass_studies_list.csv"
df = pd.read_csv(filename)

mrns = []
typeofstudys = []
label_paths = []
signal_paths = []
for i in tqdm(range(len(df))):
    path = df.Path[i]
    # replace all files path from M:\\ to Z:\\
    path = path.replace('M:\\', 'Z:\\')

    # list all files under this folder
    files = os.listdir(path)
    # find the file staring with Label and ending with .mat
    signal_path = ''
    for thisfile in files:
        if thisfile.startswith('Signal_') and thisfile.endswith('.mat'):
            signal_path = os.path.join(path, thisfile)
            break

    # list all files under this folder
    raw_path = os.path.join(path, 'raw')
    if not os.path.exists(raw_path):
        continue
    files = os.listdir(raw_path)
    label_path = ''
    for thisfile in files:
        if thisfile.startswith('Labels_') and thisfile.endswith('.mat'):
            label_path = os.path.join(path, 'raw', thisfile)
            break

    if label_path == '' or signal_path=='':
        continue
    else:
        label_paths.append(label_path)
        signal_paths.append(signal_path)
        mrns.append(df.MRN.iloc[i])
        typeofstudys.append(df.TypeOfTest.iloc[i])

df = pd.DataFrame(data={'MRN':mrns,
                        'TypeOfTest':typeofstudys,
                        'label_path':label_paths,
                        'signal_paths':signal_paths})
df.to_csv('file_paths.csv', index=False)
