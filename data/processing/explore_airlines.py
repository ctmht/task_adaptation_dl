import numpy as np
import pandas as pd
from scipy.io.arff import loadarff 

import os
path = os.path.abspath('./AIRLINES_10M.arff')
print(path)

raw_data = loadarff(path)
print(type(raw_data[0]), type(raw_data[1]), sep='\n')
for entname, enttype in zip(raw_data[1].names(), raw_data[1].types()):
    print(f"{entname : >15}, {enttype : >1}")

df_data = pd.DataFrame(raw_data[0])