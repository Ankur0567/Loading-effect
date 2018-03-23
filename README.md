# BC_loading_correction

## 1. Input

1) No. of spots need to be binned and calculate k `Spots = 5`
2) No. of bin, the more the bins, the smaller the binned ATN unit. `bin_number = 30`
3) Rawdata excel file, following the format in the provided template "`Rawdata.xlsx`"


## 2. Output

1) `Output_k.csv`, which includes k value for every tape advance
2) `Output_corrected_data.csv`, which includes corrected BC data

## 3. Bin all data grouped by tape advance

please run `Avg_all_spots.py`, you will get BC vs ATN in a csv file, 
then calculate the k with linear regression simply in Excel

### Note

This programm is based on bin algorithm correcting BC loading data (Park et al., 2010). 

>Park, S.S., A.D.A. Hansen and S.Y. Cho, Measurement of real time black carbon for investigating spot loading effects of Aethalometer data. Atmospheric Environment, 2010. 44(11): p. 1449-1455.
