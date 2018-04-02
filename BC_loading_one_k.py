import pandas as pd
import numpy as np
from sklearn import linear_model
from scipy.interpolate import UnivariateSpline
import time
######
#This program is to calculate the averaged BC vs ATN if all spots considered to be averaged
#Then you can do the linear fitting relationship in the Excel. 
######

################################## INPUT #########################################

# No. of bin, the more the bins, the smaller the binned ATN unit
bin_number = 40
# Rawdata file, following the format in the provided template "Rawdata.xlsx"
print '%d bins were set.' % bin_number
print ' Processing...'

xl_file = pd.ExcelFile("Rawdata.xlsx")

################################## Main Function ##################################
df = xl_file.parse().dropna(axis=1, how='all')
Datetime = df.dropna().Datetime
df.dropna()

### identify tape advance by Null data
null_value = df.ATN1.isnull()
gb = df.dropna().groupby(null_value.cumsum())


channels = (len(df.columns) - 1) / 2

BC = []
ATN = []
RSq = []
k=[]
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
    binTime_df = pd.DataFrame({'TapeAdvanceTime': bin_time})

    for j in xrange(0, 1):
        df_concat = pd.concat((binned_group[j:j + tape_advances]), axis=1)
        data = df_concat.groupby(level=0, axis=1).mean()

        x = data[['ATN' + str(i)]].values
        y = data['BC' + str(i)].values

        y_spl = UnivariateSpline(x, y, s=0, k=4)
        x_range = np.linspace(x[0], x[-1], len(x))

        y_spl_2d = y_spl.derivative(n=2)
        abs_y_2d = abs(y_spl_2d(x_range))

        idx = np.where(abs_y_2d > np.nanpercentile(abs_y_2d, 90))
        x = np.delete(x, idx, 0)
        y = np.delete(y, idx, 0)

        reg = linear_model.LinearRegression()
        reg.fit(x, y)
        R_square = reg.score(x, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_
        k_value = -slope / intercept

        ATN_df = data['ATN' + str(i)]
        BC_df = data['BC' + str(i)]
        ATN.append(ATN_df)
        BC.append(BC_df)
        RSq.append(R_square)
        k.append(k_value)


BC = pd.DataFrame(BC).T
ATN = pd.DataFrame(ATN).T
RSq = pd.DataFrame(RSq,columns= ['RSq1-x'])
k_df =pd.DataFrame (k,columns=['k1-x'])


tape_advance_data = pd.concat([ATN,BC], axis=1)
k_R = pd.concat([k_df,RSq], axis=1)
csv_input =  pd.concat([tape_advance_data,k_R], axis=1)
csv_input.to_csv('AllSpots_BCvsATN_Output.csv', index=False)

print k_R


#////////////////////////////Correct Raw BC data////////////////////////////#
BC=[]
BC_rawdata = []
for i in xrange(1,channels+1):
    BC_corrected = []
    BC_nc = []
    m=0
    for key, item in gb:
        spot= gb.get_group(key)
        BC_raw = spot['BC'+str(i)]
        BC_corr = spot['BC'+str(i)]/(1-spot['ATN'+str(i)]*k[i-1])
        m=m+1
        BC_corrected.append(BC_corr)
        BC_nc.append(BC_raw)
    df = pd.concat(BC_corrected).to_frame()
    df.columns = ['corrected_BC'+str(i)]
    BC.append(df)
    r_df = pd.concat(BC_nc).to_frame()
    r_df.columns = ['raw_BC'+str(i)]
    BC_rawdata.append(df)
	
corrected_BC = pd.concat(BC,axis = 1)
raw_BC = pd.concat(BC_rawdata,axis = 1)
corrected_data = pd.concat((Datetime,raw_BC,corrected_BC),axis=1)
corrected_data.to_csv('Output_corrected_data_one k.csv',index = False)
print 'Correction finished, check the output CSV file'

print 'There are %d spots in this data set' % tape_advances
print 'Finished'
time.sleep(5)

