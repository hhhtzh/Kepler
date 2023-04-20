import pandas as pd
import numpy as np
dt=pd.read_csv('/home/tbz/PycharmProjects/pythonProjectAR/diagnostics.csv')
d1=dt.columns[8:9]
print(np.array(dt[d1]))
