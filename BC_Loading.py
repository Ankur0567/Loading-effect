import pandas as pd
import numpy as np
from sklearn import linear_model

# No. of spots need to be binned and calculate k
Spots = 5
# No. of bin, the more the bins, the smaller the binned ATN unit
bin_number = 30
# Rawdata file, following the format in the provided template "Rawdata.xlsx"
xl_file = pd.ExcelFile("Rawdata.xlsx")

################################## Main Function##################################
df = xl_file.parse().dropna(axis=1, how='all')
Datetime = df.dropna().Datetime
df.dropna()
### identify tape advance by Null data
null_value = df.ATN1.isnull()
gb = df.dropna().groupby(null_value.cumsum())

#//////////////////////////Get K value/////////////////////////////#
R_array = []
k_array = []
channels = (len(df.columns) - 1) / 2
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
    if Spots > tape_advances:
        print 'Binned spots should be smaller than %d. Please reinput' % (tape_advances)
        break

    startSpots = Spots / 2 + 1
    binTime_df = pd.DataFrame({'TapeAdvanceTime': bin_time})

    RSq = []
    k = []
    for j in xrange(0, tape_advances - Spots / 2 * 2):
        df_concat = pd.concat((binned_group[j:j + Spots / 2 * 2]), axis=1)
        data = df_concat.groupby(level=0, axis=1).mean()
        reg = linear_model.LinearRegression()
        reg.fit(data[['ATN' + str(i)]], data['BC' + str(i)])
        R_square = reg.score(data[['ATN' + str(i)]], data['BC' + str(i)])
        slope = reg.coef_[0]
        intercept = reg.intercept_
        k_value = -slope / intercept
        RSq.append(R_square)
        k.append(k_value)
    RSq = [RSq[0]] * (Spots / 2) + RSq + [RSq[-1]] * (Spots / 2)
    k = [k[0]] * (Spots / 2) + k + [k[-1]] * (Spots / 2)
    R_array = np.append(R_array, RSq, axis=0)
    k_array = np.append(k_array, k, axis=0)

R_array = np.reshape(R_array, (channels, tape_advances))
k_array = np.reshape(k_array, (channels, tape_advances))
R_df = pd.DataFrame(R_array.T)
R_df.columns = ['Rsq1', 'Rsq2', 'Rsq3', 'Rsq4', 'Rsq5', 'Rsq6', 'Rsq7']
k_df = pd.DataFrame(k_array.T)
k_df.columns = ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7']
tape_advance_data = pd.concat([binTime_df, k_df, R_df], axis=1)
tape_advance_data.to_csv('Output_k.csv', index=False)

#////////////////////////////Correct Raw BC data////////////////////////////#
BC=[]
for i in xrange(1,channels+1):
    BC_corrected = []
    m=0
    for key, item in gb:
        spot= gb.get_group(key)
        BC_corr = (1+spot['ATN'+str(i)]*k_array[i-1][m])*spot['BC'+str(i)]
        m=m+1
        BC_corrected.append(BC_corr)
    df = pd.concat(BC_corrected).to_frame()
    df.columns = ['BC'+str(i)]
    BC.append(df)
corrected_BC = pd.concat(BC,axis = 1)
corrected_data = pd.concat((Datetime,corrected_BC),axis=1)
corrected_data.to_csv('Output_corrected_data.csv',index = False)
print 'Correction finished, check the output CSV file'