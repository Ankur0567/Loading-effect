# BC_loading_correction

## 1. Input

1) No. of spots need to be binned and calculate k 
2) No. of bin, the more the bins, the smaller the binned ATN unit.`
3) rawdata excel file, following the format in the provided template "`template.xlsx`"


## 2. Output
1) For bin method, `Bin_k.csv` includes k value for every tape advance, `Bin_corrected_data.csv` includes corrected BC data
2) For gap method, `Gap_k.csv` includes k value for every tape advance, `Gap_corrected_data.csv` includes corrected BC data

### Note

 If the input number of spots is greater than or equal the maximum spots (tape advance), there will be one filter parameter k for BC at each wavelength. 
 And you will get the binned BC vs ATN in `Bin_k.csv`.
 
 
## Reference
 
 The bin method is based on research by Park et al. (2010), the gap method is based on study by Virkkula et al. (2007)

>Park, S.S., A.D.A. Hansen and S.Y. Cho, Measurement of real time black carbon for investigating spot loading effects of Aethalometer data. Atmospheric Environment, 2010. 44(11): p. 1449-1455.
>Virkkula, A., et al., A Simple Procedure for Correcting Loading Effects of Aethalometer Data. Journal of the Air & Waste Management Association, 2007. 57(10): p. 1214-1222.