import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.interpolate import UnivariateSpline
import time
################################## INPUT #########################################
# No. of spots need to be binned and calculate k
Spots = 15
# No. of bin, the more the bins, the smaller the binned ATN unit
bin_number = 60
# Rawdata file, following the format in the provided template "Rawdata.xlsx"
print 'Spots is %d, %d bins.' % (Spots, bin_number)
print ' Processing...'
xl_file = pd.ExcelFile("Rawdata.xlsx")

################################## Main Function ##################################
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
        bins = np.linspace(0, df['ATN' + str(i)].max(), bin_number)
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

        x = data[['ATN' + str(i)]].values
        y = data['BC' + str(i)].values


        y_spl = UnivariateSpline(x, y, s=0, k=4)
        x_range = np.linspace(x[0], x[-1], len(x))

        y_spl_2d = y_spl.derivative(n=2)
        abs_y_2d = abs(y_spl_2d(x_range))

        idx = np.where(abs_y_2d > np.nanpercentile(abs_y_2d,90))
        x = np.delete(x, idx, 0)
        y = np.delete(y,idx,0)


        reg = linear_model.LinearRegression()
        reg.fit(x, y)
        R_square = reg.score(x, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        k_value = -slope / intercept
        RSq.append(R_square)
        k.append(k_value)


    RSq = [RSq[0]] * (Spots / 2) + RSq + [RSq[-1]] * (Spots / 2)
    k = [k[0]] * (Spots / 2) + k + [k[-1]] * (Spots / 2)
    R_array = np.append(R_array, RSq, axis=0)
    k_array = np.append(k_array, k, axis=0)

column_R = []
column_k = []
for i in xrange(1,channels+1):
    column_R.append('Rsq'+str(i))
    column_k.append('k' + str(i))
R_array = np.reshape(R_array, (channels, tape_advances))
k_array = np.reshape(k_array, (channels, tape_advances))
R_df = pd.DataFrame(R_array.T)
R_df.columns = column_R
k_df = pd.DataFrame(k_array.T)
k_df.columns = column_k
tape_advance_data = pd.concat([binTime_df, k_df, R_df], axis=1)
tape_advance_data.to_csv('Output_moving k.csv', index=False)

#////////////////////////////Correct Raw BC data////////////////////////////#
BC=[]
BC_rawdata = []
for i in xrange(1,channels+1):
    BC_corrected = []
    BC_nc = []
    m=0
    for key, item in gb:
        spot= gb.get_group(key)
        BC_raw = spot['BC' + str(i)]
        BC_corr = spot['BC'+str(i)]/(1-spot['ATN'+str(i)]*k_array[i-1][m])
        m=m+1
        BC_corrected.append(BC_corr)
        BC_nc.append(BC_raw)
    df = pd.concat(BC_corrected).to_frame()
    df.columns = ['corrected_BC'+str(i)]
    BC.append(df)
    r_df = pd.concat(BC_nc).to_frame()
    r_df.columns = ['raw_BC' + str(i)]
    BC_rawdata.append(r_df)
corrected_BC = pd.concat(BC,axis = 1)
raw_BC = pd.concat(BC_rawdata,axis = 1)
corrected_data = pd.concat((Datetime,raw_BC,corrected_BC),axis=1)
corrected_data.to_csv('Output_corrected_data_moving k.csv',index = False)
print 'Correction finished, check the output CSV file'
time.sleep(5)
