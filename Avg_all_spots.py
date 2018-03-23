import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.interpolate import UnivariateSpline
from operator import itemgetter
from itertools import groupby
######
#This program is to calculate the averaged BC vs ATN if all spots considered to be averaged
#Then you can do the linear fitting relationship in the Excel. 
######

################################## INPUT #########################################

# No. of bin, the more the bins, the smaller the binned ATN unit
bin_number = 60
# Rawdata file, following the format in the provided template "Rawdata.xlsx"
xl_file = pd.ExcelFile("Rawdata.xlsx")

################################## Main Function ##################################
df = xl_file.parse().dropna(axis=1, how='all')
Datetime = df.dropna().Datetime
df.dropna()

### identify tape advance by Null data
null_value = df.ATN1.isnull()
gb = df.dropna().groupby(null_value.cumsum())

R_array = []
k_array = []
channels = (len(df.columns) - 1) / 2

BC = []
ATN = []
for i in xrange(1, channels + 1):

    binned_group = []
    bin_time = []
    for key, item in gb:
        spot = gb.get_group(key)
        bins = np.linspace(0, spot['ATN' + str(i)].max(), bin_number)
        groups = spot.groupby(np.digitize(spot['ATN' + str(i)], bins))
        binned_group.append(groups.mean())
        bin_time.append(spot['Datetime'].iloc[0])
    tape_advances = len(binned_group)   
    binTime_df = pd.DataFrame({'TapeAdvanceTime': bin_time})

    for j in xrange(0, 1):
        df_concat = pd.concat((binned_group[j:j + tape_advances]), axis=1)
        data = df_concat.groupby(level=0, axis=1).mean()

        x = data['ATN' + str(i)]
        y = data['BC' + str(i)]
    BC.append(y)
    ATN.append(x)
BC = pd.DataFrame(BC)
ATN = pd.DataFrame(ATN)

tape_advance_data = pd.concat([BC, ATN], axis=0).T
tape_advance_data.to_csv('AllSpots_BCvsATN_Output.csv', index=False)
