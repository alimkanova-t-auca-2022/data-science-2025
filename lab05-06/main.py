import pandas as pd
import seaborn as sbr
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

df1 = pd.read_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\API_SH.STA.AIRP.MA.P5_DS2_en_excel_v2_3403.xls", sheet_name="Data", skiprows=3)
df2 = pd.read_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\API_SH.PRV.SMOK_DS2_en_excel_v2_3887.xls", sheet_name="Data",skiprows=3)
df3 = pd.read_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\API_EN.ATM.PM25.MC.ZS_DS2_en_excel_v2_3588.xls", sheet_name="Data", skiprows=3)
df4 = pd.read_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\API_SH.DYN.NCOM.ZS_DS2_en_excel_v2_3058.xls", sheet_name="Data", skiprows=3)
df5 = pd.read_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\API_NY.GDP.MKTP.KD.ZG_DS2_en_excel_v2_67.xls", sheet_name="Data", skiprows=3)
df6 = pd.read_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\API_NV.IND.TOTL.KD.ZG_DS2_en_excel_v2_2566.xls", sheet_name="Data", skiprows=3)
df7 = pd.read_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\API_ER.GDP.FWTL.M3.KD_DS2_en_excel_v2_3500.xls", sheet_name="Data", skiprows=3)

def extract_canada(df, column_name, country='Canada'):
    country_row = df[df['Country Name'] == country]
    years_data = country_row.iloc[0,4:]
    years_data.index = years_data.index.astype(str)
    valid_years = [y for y in years_data.index if str(y).isdigit()]

    years = [int(y) for y in valid_years]
    values = years_data[valid_years].values.tolist()
    return pd.DataFrame({
        'year': years,
        column_name: values
        })

df1 = extract_canada(df1, 'mortality_rate')
df2 = extract_canada(df2, 'smoking_prevalence')
df3 = extract_canada(df3, 'air_pollution')
df4 = extract_canada(df4, 'mortality')
df5 = extract_canada(df5, 'gdp_growth')
df6 = extract_canada(df6, 'industry')
df7 = extract_canada(df7, 'water_productivity')
df_canada = df1.merge(df2, on='year', how='outer')
df_canada = df_canada.merge(df3, on='year', how='outer')
df_canada = df_canada.merge(df4, on='year', how='outer')
df_canada = df_canada.merge(df5, on='year', how='outer')
df_canada = df_canada.merge(df6, on='year', how='outer')
df_canada = df_canada.merge(df7, on='year', how='outer')

df_canada.to_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\Canada_Raw_Data.xlsx", index=False)
print(df_canada.head())

##
df_vis = df_canada.set_index('year')
plt.figure(figsize=(12,6))
sbr.heatmap(df_vis.isnull(), cmap="crest")
plt.title("Heatmap of Missing Values")
plt.xticks(rotation=0, ha='center') 
plt.yticks(rotation=0)   
plt.tight_layout()
plt.show()

calcul_perc = df_vis.isnull().mean() *100
print("The percentage of missing values:\n", calcul_perc)

row_threshold = int(len(df_canada.columns) * 0.35)
column_threshold = int(len(df_canada) * 0.35)
canada_filtered =  df_canada.dropna(axis=1, thresh=column_threshold)\
                          .dropna(thresh=row_threshold, axis=0)

plt.figure(figsize=(12, 6))
sbr.heatmap(canada_filtered.set_index('year').isnull(), cmap="crest")
plt.title("Heatmap of Missing Values (Filtered Data)")
plt.xticks(rotation=0, ha='center') 
plt.yticks(rotation=0)  
plt.tight_layout()
plt.show()

calcul_perc_filt = canada_filtered.isnull().mean() * 100
print("\nPercentage of missing values after filtering:\n", calcul_perc_filt)

canada_filtered.to_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\Canada_Cleaned.xlsx", index=False)
##

features_with_missing = canada_filtered.columns[canada_filtered.isnull().any()]
original_missing_cols = [col for col in df_canada.columns
                        if df_canada[col].isnull().any() and col != 'year']

print("Features with missing values: \n",features_with_missing)

for col in original_missing_cols:
    if col in canada_filtered.columns:
        plt.figure(figsize=(12, 4))
        plt.plot(canada_filtered['year'], canada_filtered[col], 'o-')
        plt.title(col)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def impute_data(series):
    if series.count() < 5:
        return series.fillna(series.mean())
    
    try:
        interpol = series.interpolate(method='time')
        if interpol.isnull().any():
            decomposition = seasonal_decompose(interpol.ffill().bfill(), model = 'additive', period=1, extrapolate_trend='freq')
            trend = decomposition.trend
            return series.fillna(trend)
    except:
        return series.ffill().bfill()
    
for col in canada_filtered:
    if col !='year':
        canada_filtered[col] = impute_data(canada_filtered[col])

canada_filtered.to_excel("C:\\Users\\User\D\загрузкии\Data Science Labs\data-science-2025\lab05-06\Canada_Completed.xlsx",index=False)




