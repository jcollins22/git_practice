
# Kings County Data

We were given a dataset regarding housing prices in Kings County. Our task was to create a model that predicts selling price of a house.


```python
#import data file
import pandas as pd
import numpy as np
df = pd.read_csv("kc_house_data.csv")
```

## First Import and Clean Data

After importing the needed packages and exploring the data, it was clear that cleaning was needed. Specifically, there were placeholder symbols in some columns, some columns were the wrong data type and some had NaN values that needed to be replaced. After taking care of these things, there were features to engineer. In this case, I created an value for the age of the house, how much of the square feet was not in the basement, bathrooms per bedroom, and a boolean value for whether the house had a basement. 


```python
df["sqft_basement"].replace(to_replace="?", value="0.0", inplace=True) #fix placeholders
df["sqft_basement"] = df["sqft_basement"].astype(float, inplace=True) #reassign to float
df["basement"] = df["sqft_basement"].apply(lambda x: False if x == 0.0 else True) #create a boolean variable
df["yr_renovated"].fillna(value = 0) #deal with missing data
df["renovated"] = df["yr_renovated"].map(lambda x: x > 0, True) #boolean feature
df["age_of_house"] = 2019 - df["yr_built"] #engineer feature
df["upstairs_as_percent_of_house"] = (df["sqft_living"] - df["sqft_basement"]) / df["sqft_living"] #new feature
df["bath_per_bed"] = df["bathrooms"] / df["bedrooms"] #new feature
```

## Next Coding Was Needed

At this point I still needed a way to deal with zipcodes and I also needed a way to create 1 hot encoding for ordinal and categorical variables. For the zipcode data, I did outside research and grouped the zipcodes by high, midhigh, and mid average income for the area. Keep in mind, the mid is still a large average income as it is a wealthy area on the whole.  


```python
#creating helpful lists
high_price_zipcodes = [98039, 98040, 98004, 98112]
midhigh_price_zipcodes = [98075, 98033, 98074, 98053, 98121, 98006, 98199]
mid_price_zipcodes = [98105, 98065, 98177, 98005, 98052, 98029, 98119, 98027 ,98072]
top_20_zipcodes = high_price_zipcodes + midhigh_price_zipcodes + mid_price_zipcodes

#creating a zipcode rank feature
zipcode_rank = []

for zipcode in df.zipcode:
    if zipcode in set(high_price_zipcodes):
        zipcode_rank.append("high")        
for zipcode in df.zipcode:
    if zipcode in set(midhigh_price_zipcodes):
        zipcode_rank.append("midhigh")       
for zipcode in df.zipcode:
    if zipcode in set(mid_price_zipcodes):
        zipcode_rank.append("mid")
for zipcode in df.zipcode:
    if zipcode not in set(top_20_zipcodes):
        zipcode_rank.append("other")

df["zipcode_rank"] = zipcode_rank

#recoding for final transform
zipcode_recode = []
for i in df["zipcode_rank"]:
    if i == "high":
        zipcode_recode.append(3)
    elif i == "midhigh":
        zipcode_recode.append(2)
    elif i == "mid":
        zipcode_recode.append(1)
    else:
        zipcode_recode.append(0)
df["zipcode_recode"] = zipcode_recode
```

I then created the 1 hot encoding columns for all ordinal and categorical variables.


```python
#making dummy variables

bathroom_bins = [0, 2, 4, 8]
df["bath_bin"] = pd.cut(df["bathrooms"], bathroom_bins)
df.bath_bin.value_counts()

bedroom_bins = [0, 2, 3, 5, 33]
df["bed_bin"] = pd.cut(df["bedrooms"], bedroom_bins)
df.bed_bin.value_counts()

bed_dummy = pd.get_dummies(df.bed_bin, prefix="BED")
bath_dummy = pd.get_dummies(df.bath_bin, prefix="BATH")
df = pd.concat([df, bed_dummy, bath_dummy], axis=1)

condition_bin = [0, 1, 2, 3, 4, 5]
df["condition_bin"] = pd.cut(df["condition"], condition_bin)
condition_dummy = pd.get_dummies(df.condition_bin, prefix="COND")
df = pd.concat([df, condition_dummy], axis=1)

floor_bins = [0, 1, 2, 3, 4]
df["floor_bin"] = pd.cut(df["floors"], floor_bins)
floor_dummy = pd.get_dummies(df.floor_bin, prefix="FLOORS")
df = pd.concat([df, floor_dummy], axis=1)

zip_bin = [0, 1, 2, 3]
df["zip_bin"] = pd.cut(df["zipcode_recode"], zip_bin)
zip_dummy = pd.get_dummies(df.zip_bin, prefix="ZIP")
df = pd.concat([df, zip_dummy], axis=1)

view_bins = [0, 1, 2, 3, 4, 5]
df["view_bin"] = pd.cut(df["view"], view_bins)
view_dummy = pd.get_dummies(df.view_bin, prefix="VIEW")
df = pd.concat([df, view_dummy], axis=1)

grade_bins = [1, 3, 7, 11, 13]
df["grade_bin"] = pd.cut(df["grade"], grade_bins)
grade_dummy = pd.get_dummies(df.grade_bin, prefix="GRADE")
df = pd.concat([df, grade_dummy], axis=1)


```

## These Continuous Variables Aren't Normal

At this point we have all categorical and orginal variables recoded and now we need to perform transforms on the continuous data to help normalize it. To do this i used log transforms on all data except the age of the house. I was able to use min/max scaling for age of the house for a better fit for normality.


```python
#variable transforms for normality

df["log_sqft_living15"] = np.log(df.sqft_living15)
df['log_sqft_lot15'] = np.log(df.sqft_lot15)
x_age = df.age_of_house
df["scale_age_of_house"] = (x_age - x_age.min()) / (x_age.max() - x_age.min())
df["log_sqft_living"] = np.log(df.sqft_living)
df["log_sqft_lot"] = np.log(df.sqft_lot)

```

## Drop the Columns Like They Are Hot

Our data frame is now much larger than it needs to be. First of all, we must drop the first column of all 1 hot encoding variables to prevent multicolinearity as well as the initial column the data was encoded from. Then we need to drop other colums that show high degree of multicolinearity.


```python
#drop unneeded columns including the first colums of 1-hot encoded variables
df_clean = df.drop(columns = ['id', 'date', 'bedrooms', 'bathrooms', 'sqft_living',
       'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
       'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
       'lat', 'long', 'sqft_living15', 'sqft_lot15', 'age_of_house', 
       'upstairs_as_percent_of_house','zipcode_rank', 'zipcode_recode', 
       'bath_bin', 'bed_bin', 'BED_(0, 2]', 'BATH_(0, 2]', 'condition_bin', 
       'COND_(0, 1]', 'floor_bin', 'FLOORS_(0, 1]', 'zip_bin', 'view_bin', 
       'VIEW_(0, 1]', 'grade_bin', 'GRADE_(1, 3]'])
```

## Remove Outliers for Better Modeling

Now we need to remove the outliers, create and verify the model. First I removed all houses over 7 million dollars. Then the appropriate libraries for modeling were imported and two models were created. The first used all of the remaining columns of the data frame. The second removed a few variables that were on the boarder of our cuttoff for multicolinearity. 


```python
#remove price outliers
df_clean = df_clean[df_clean.price < 7000000]
```


```python
#prepare for modeling 1
features = df_clean.drop(columns = "price")
target = df_clean["price"]
X1 = features
y1 = target

#1st Model
import statsmodels.api as sm
X_int_sm1 = sm.add_constant(X1)
model1 = sm.OLS(y1.astype(float), X_int_sm1.astype(float)).fit()
model1.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.593</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.592</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1120.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 31 May 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>18:14:13</td>     <th>  Log-Likelihood:    </th> <td>-2.9732e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21595</td>      <th>  AIC:               </th>  <td>5.947e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21566</td>      <th>  BIC:               </th>  <td>5.949e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    28</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>basement</th>           <td>-3.681e+06</td> <td> 2.41e+05</td> <td>  -15.279</td> <td> 0.000</td> <td>-4.15e+06</td> <td>-3.21e+06</td>
</tr>
<tr>
  <th>renovated</th>          <td> 5.413e+04</td> <td> 9124.851</td> <td>    5.932</td> <td> 0.000</td> <td> 3.62e+04</td> <td>  7.2e+04</td>
</tr>
<tr>
  <th>bath_per_bed</th>       <td>  1.02e+05</td> <td> 1.28e+04</td> <td>    7.946</td> <td> 0.000</td> <td> 7.69e+04</td> <td> 1.27e+05</td>
</tr>
<tr>
  <th>BED_(2, 3]</th>         <td>-6.627e+04</td> <td> 6073.926</td> <td>  -10.911</td> <td> 0.000</td> <td>-7.82e+04</td> <td>-5.44e+04</td>
</tr>
<tr>
  <th>BED_(3, 5]</th>         <td>-7.297e+04</td> <td> 8418.807</td> <td>   -8.667</td> <td> 0.000</td> <td>-8.95e+04</td> <td>-5.65e+04</td>
</tr>
<tr>
  <th>BED_(5, 33]</th>        <td>-1.167e+05</td> <td> 1.66e+04</td> <td>   -7.043</td> <td> 0.000</td> <td>-1.49e+05</td> <td>-8.42e+04</td>
</tr>
<tr>
  <th>BATH_(2, 4]</th>        <td>-2.841e+04</td> <td> 5566.929</td> <td>   -5.103</td> <td> 0.000</td> <td>-3.93e+04</td> <td>-1.75e+04</td>
</tr>
<tr>
  <th>BATH_(4, 8]</th>        <td> 4.384e+05</td> <td> 1.74e+04</td> <td>   25.238</td> <td> 0.000</td> <td> 4.04e+05</td> <td> 4.72e+05</td>
</tr>
<tr>
  <th>COND_(1, 2]</th>        <td> 6.617e+04</td> <td> 4.65e+04</td> <td>    1.424</td> <td> 0.154</td> <td>-2.49e+04</td> <td> 1.57e+05</td>
</tr>
<tr>
  <th>COND_(2, 3]</th>        <td> 8.475e+04</td> <td> 4.31e+04</td> <td>    1.965</td> <td> 0.049</td> <td>  225.486</td> <td> 1.69e+05</td>
</tr>
<tr>
  <th>COND_(3, 4]</th>        <td> 9.691e+04</td> <td> 4.31e+04</td> <td>    2.247</td> <td> 0.025</td> <td> 1.24e+04</td> <td> 1.81e+05</td>
</tr>
<tr>
  <th>COND_(4, 5]</th>        <td> 1.435e+05</td> <td> 4.34e+04</td> <td>    3.307</td> <td> 0.001</td> <td> 5.85e+04</td> <td> 2.29e+05</td>
</tr>
<tr>
  <th>FLOORS_(1, 2]</th>      <td> 2.148e+04</td> <td> 3946.692</td> <td>    5.444</td> <td> 0.000</td> <td> 1.37e+04</td> <td> 2.92e+04</td>
</tr>
<tr>
  <th>FLOORS_(2, 3]</th>      <td>  1.43e+05</td> <td> 9660.350</td> <td>   14.798</td> <td> 0.000</td> <td> 1.24e+05</td> <td> 1.62e+05</td>
</tr>
<tr>
  <th>FLOORS_(3, 4]</th>      <td> 2.053e+05</td> <td> 8.76e+04</td> <td>    2.344</td> <td> 0.019</td> <td> 3.36e+04</td> <td> 3.77e+05</td>
</tr>
<tr>
  <th>ZIP_(0, 1]</th>         <td> -245.5383</td> <td> 4815.518</td> <td>   -0.051</td> <td> 0.959</td> <td>-9684.309</td> <td> 9193.233</td>
</tr>
<tr>
  <th>ZIP_(1, 2]</th>         <td>-4335.7065</td> <td> 5037.513</td> <td>   -0.861</td> <td> 0.389</td> <td>-1.42e+04</td> <td> 5538.192</td>
</tr>
<tr>
  <th>ZIP_(2, 3]</th>         <td>-1.246e+04</td> <td> 7850.836</td> <td>   -1.587</td> <td> 0.112</td> <td>-2.78e+04</td> <td> 2927.642</td>
</tr>
<tr>
  <th>VIEW_(1, 2]</th>        <td> 6.817e+04</td> <td> 7827.171</td> <td>    8.710</td> <td> 0.000</td> <td> 5.28e+04</td> <td> 8.35e+04</td>
</tr>
<tr>
  <th>VIEW_(2, 3]</th>        <td> 1.525e+05</td> <td> 1.06e+04</td> <td>   14.337</td> <td> 0.000</td> <td> 1.32e+05</td> <td> 1.73e+05</td>
</tr>
<tr>
  <th>VIEW_(3, 4]</th>        <td> 5.137e+05</td> <td> 1.35e+04</td> <td>   38.104</td> <td> 0.000</td> <td> 4.87e+05</td> <td>  5.4e+05</td>
</tr>
<tr>
  <th>VIEW_(4, 5]</th>        <td>   1.7e-08</td> <td> 7.93e-09</td> <td>    2.143</td> <td> 0.032</td> <td> 1.45e-09</td> <td> 3.25e-08</td>
</tr>
<tr>
  <th>GRADE_(3, 7]</th>       <td> -8.45e+04</td> <td> 2.31e+05</td> <td>   -0.366</td> <td> 0.715</td> <td>-5.37e+05</td> <td> 3.68e+05</td>
</tr>
<tr>
  <th>GRADE_(7, 11]</th>      <td> 2.106e+04</td> <td> 2.31e+05</td> <td>    0.091</td> <td> 0.927</td> <td>-4.32e+05</td> <td> 4.74e+05</td>
</tr>
<tr>
  <th>GRADE_(11, 13]</th>     <td>  1.01e+06</td> <td> 2.32e+05</td> <td>    4.346</td> <td> 0.000</td> <td> 5.55e+05</td> <td> 1.47e+06</td>
</tr>
<tr>
  <th>log_sqft_living15</th>  <td> 2.184e+05</td> <td> 7896.729</td> <td>   27.656</td> <td> 0.000</td> <td> 2.03e+05</td> <td> 2.34e+05</td>
</tr>
<tr>
  <th>log_sqft_lot15</th>     <td>-1.636e+04</td> <td> 4985.585</td> <td>   -3.282</td> <td> 0.001</td> <td>-2.61e+04</td> <td>-6591.247</td>
</tr>
<tr>
  <th>scale_age_of_house</th> <td> 3.068e+05</td> <td> 8724.034</td> <td>   35.169</td> <td> 0.000</td> <td>  2.9e+05</td> <td> 3.24e+05</td>
</tr>
<tr>
  <th>log_sqft_living</th>    <td> 3.492e+05</td> <td> 7932.288</td> <td>   44.017</td> <td> 0.000</td> <td> 3.34e+05</td> <td> 3.65e+05</td>
</tr>
<tr>
  <th>log_sqft_lot</th>       <td>-1.349e+04</td> <td> 4525.209</td> <td>   -2.981</td> <td> 0.003</td> <td>-2.24e+04</td> <td>-4620.277</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12545.791</td> <th>  Durbin-Watson:     </th>  <td>   1.978</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>319456.464</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.316</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>21.264</td>   <th>  Cond. No.          </th>  <td>1.00e+16</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 5.98e-26. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
#prepare for modeling 2
features2 = features.drop(columns = ["basement", "VIEW_(4, 5]", "log_sqft_living15", "log_sqft_lot15"])
target2 = df_clean["price"]
X2 = features2
y2 = target2

#1st Model
import statsmodels.api as sm
X_int_sm2 = sm.add_constant(X2)
model2 = sm.OLS(y2.astype(float), X_int_sm2.astype(float)).fit()
model2.summary()

```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>price</td>      <th>  R-squared:         </th>  <td>   0.578</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.578</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   1137.</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 31 May 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>17:59:43</td>     <th>  Log-Likelihood:    </th> <td>-2.9769e+05</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td> 21595</td>      <th>  AIC:               </th>  <td>5.954e+05</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td> 21568</td>      <th>  BIC:               </th>  <td>5.957e+05</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    26</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
           <td></td>             <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>              <td>-2.758e+06</td> <td> 2.43e+05</td> <td>  -11.365</td> <td> 0.000</td> <td>-3.23e+06</td> <td>-2.28e+06</td>
</tr>
<tr>
  <th>renovated</th>          <td> 4.159e+04</td> <td> 9268.135</td> <td>    4.487</td> <td> 0.000</td> <td> 2.34e+04</td> <td> 5.98e+04</td>
</tr>
<tr>
  <th>bath_per_bed</th>       <td> 8.903e+04</td> <td> 1.31e+04</td> <td>    6.821</td> <td> 0.000</td> <td> 6.34e+04</td> <td> 1.15e+05</td>
</tr>
<tr>
  <th>BED_(2, 3]</th>         <td>-7.663e+04</td> <td> 6168.685</td> <td>  -12.422</td> <td> 0.000</td> <td>-8.87e+04</td> <td>-6.45e+04</td>
</tr>
<tr>
  <th>BED_(3, 5]</th>         <td> -8.46e+04</td> <td> 8554.717</td> <td>   -9.889</td> <td> 0.000</td> <td>-1.01e+05</td> <td>-6.78e+04</td>
</tr>
<tr>
  <th>BED_(5, 33]</th>        <td>-1.533e+05</td> <td> 1.68e+04</td> <td>   -9.127</td> <td> 0.000</td> <td>-1.86e+05</td> <td> -1.2e+05</td>
</tr>
<tr>
  <th>BATH_(2, 4]</th>        <td>-1.816e+04</td> <td> 5651.980</td> <td>   -3.213</td> <td> 0.001</td> <td>-2.92e+04</td> <td>-7079.156</td>
</tr>
<tr>
  <th>BATH_(4, 8]</th>        <td> 4.426e+05</td> <td> 1.77e+04</td> <td>   25.041</td> <td> 0.000</td> <td> 4.08e+05</td> <td> 4.77e+05</td>
</tr>
<tr>
  <th>COND_(1, 2]</th>        <td> 2.366e+04</td> <td> 4.72e+04</td> <td>    0.501</td> <td> 0.616</td> <td>-6.89e+04</td> <td> 1.16e+05</td>
</tr>
<tr>
  <th>COND_(2, 3]</th>        <td> 4.301e+04</td> <td> 4.38e+04</td> <td>    0.981</td> <td> 0.327</td> <td>-4.29e+04</td> <td> 1.29e+05</td>
</tr>
<tr>
  <th>COND_(3, 4]</th>        <td> 5.282e+04</td> <td> 4.38e+04</td> <td>    1.205</td> <td> 0.228</td> <td>-3.31e+04</td> <td> 1.39e+05</td>
</tr>
<tr>
  <th>COND_(4, 5]</th>        <td> 9.306e+04</td> <td> 4.41e+04</td> <td>    2.110</td> <td> 0.035</td> <td> 6628.452</td> <td> 1.79e+05</td>
</tr>
<tr>
  <th>FLOORS_(1, 2]</th>      <td> 2.266e+04</td> <td> 4012.936</td> <td>    5.646</td> <td> 0.000</td> <td> 1.48e+04</td> <td> 3.05e+04</td>
</tr>
<tr>
  <th>FLOORS_(2, 3]</th>      <td> 1.211e+05</td> <td> 9781.771</td> <td>   12.378</td> <td> 0.000</td> <td> 1.02e+05</td> <td>  1.4e+05</td>
</tr>
<tr>
  <th>FLOORS_(3, 4]</th>      <td> 1.769e+05</td> <td> 8.91e+04</td> <td>    1.986</td> <td> 0.047</td> <td> 2304.291</td> <td> 3.52e+05</td>
</tr>
<tr>
  <th>ZIP_(0, 1]</th>         <td> 1779.6280</td> <td> 4899.351</td> <td>    0.363</td> <td> 0.716</td> <td>-7823.462</td> <td> 1.14e+04</td>
</tr>
<tr>
  <th>ZIP_(1, 2]</th>         <td>-4330.7444</td> <td> 5125.707</td> <td>   -0.845</td> <td> 0.398</td> <td>-1.44e+04</td> <td> 5716.020</td>
</tr>
<tr>
  <th>ZIP_(2, 3]</th>         <td>-1.097e+04</td> <td> 7988.046</td> <td>   -1.373</td> <td> 0.170</td> <td>-2.66e+04</td> <td> 4686.064</td>
</tr>
<tr>
  <th>VIEW_(1, 2]</th>        <td> 8.486e+04</td> <td> 7940.742</td> <td>   10.687</td> <td> 0.000</td> <td> 6.93e+04</td> <td>    1e+05</td>
</tr>
<tr>
  <th>VIEW_(2, 3]</th>        <td>  1.75e+05</td> <td> 1.08e+04</td> <td>   16.215</td> <td> 0.000</td> <td> 1.54e+05</td> <td> 1.96e+05</td>
</tr>
<tr>
  <th>VIEW_(3, 4]</th>        <td> 5.379e+05</td> <td> 1.37e+04</td> <td>   39.313</td> <td> 0.000</td> <td> 5.11e+05</td> <td> 5.65e+05</td>
</tr>
<tr>
  <th>GRADE_(3, 7]</th>       <td>-1.479e+05</td> <td> 2.35e+05</td> <td>   -0.629</td> <td> 0.529</td> <td>-6.09e+05</td> <td> 3.13e+05</td>
</tr>
<tr>
  <th>GRADE_(7, 11]</th>      <td>-1.536e+04</td> <td> 2.35e+05</td> <td>   -0.065</td> <td> 0.948</td> <td>-4.76e+05</td> <td> 4.46e+05</td>
</tr>
<tr>
  <th>GRADE_(11, 13]</th>     <td> 9.957e+05</td> <td> 2.37e+05</td> <td>    4.209</td> <td> 0.000</td> <td> 5.32e+05</td> <td> 1.46e+06</td>
</tr>
<tr>
  <th>scale_age_of_house</th> <td> 2.956e+05</td> <td> 8833.694</td> <td>   33.458</td> <td> 0.000</td> <td> 2.78e+05</td> <td> 3.13e+05</td>
</tr>
<tr>
  <th>log_sqft_living</th>    <td>  4.42e+05</td> <td> 7312.312</td> <td>   60.452</td> <td> 0.000</td> <td> 4.28e+05</td> <td> 4.56e+05</td>
</tr>
<tr>
  <th>log_sqft_lot</th>       <td>-1.517e+04</td> <td> 2091.430</td> <td>   -7.255</td> <td> 0.000</td> <td>-1.93e+04</td> <td>-1.11e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>12233.677</td> <th>  Durbin-Watson:     </th>  <td>   1.977</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th>  <td> 0.000</td>   <th>  Jarque-Bera (JB):  </th> <td>290691.836</td>
</tr>
<tr>
  <th>Skew:</th>           <td> 2.256</td>   <th>  Prob(JB):          </th>  <td>    0.00</td> 
</tr>
<tr>
  <th>Kurtosis:</th>       <td>20.398</td>   <th>  Cond. No.          </th>  <td>3.51e+03</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 3.51e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



## Data Modled with Scikit Learn and MSE and MSRE Calculated

I also created models with each of the two data frames with scikit learn to use the cross validation option. Then the mean squared error and the mean square root error were calulated for each model.


```python
#Model 1.2
from sklearn.model_selection import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.20)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train1, y_train1)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
y_hat_train1 = linreg.predict(X_train1)
y_hat_test1 = linreg.predict(X_test1)
train_residuals1 = y_hat_train1 - y_train1
test_residuals1 = y_hat_test1 - y_test1
from sklearn.metrics import mean_squared_error
train_mse1 = mean_squared_error(y_train1, y_hat_train1)
test_mse1 = mean_squared_error(y_test1, y_hat_test1)
train_MSRE1 = np.sqrt(train_mse1)
test_MSRE1 = np.sqrt(test_mse1)
print('Train Mean Squarred Error1:', train_mse1)
print('Test Mean Squarred Error1:', test_mse1)
print('Train Mean Square Root Error1', train_MSRE1)
print('Test Mean Square Root Error1', test_MSRE1)
```

    Train Mean Squarred Error1: 54516557076.60007
    Test Mean Squarred Error1: 48294342892.66986
    Train Mean Square Root Error1 233487.8092676362
    Test Mean Square Root Error1 219759.7390166585



```python
#Model 2.2

from sklearn.model_selection import train_test_split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size = 0.20)
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train2, y_train2)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
y_hat_train2 = linreg.predict(X_train2)
y_hat_test2 = linreg.predict(X_test2)
train_residuals2 = y_hat_train2 - y_train2
test_residuals2 = y_hat_test2 - y_test2
from sklearn.metrics import mean_squared_error
train_mse2 = mean_squared_error(y_train2, y_hat_train2)
test_mse2 = mean_squared_error(y_test2, y_hat_test2)
train_MSRE2 = np.sqrt(train_mse2)
test_MSRE2 = np.sqrt(test_mse2)
print('Train Mean Squarred Error2:', train_mse2)
print('Test Mean Squarred Error2:', test_mse2)
print('Train Mean Square Root Error2', train_MSRE2)
print('Test Mean Square Root Error2', test_MSRE2)
```

    Train Mean Squarred Error2: 54191761485.7682
    Test Mean Squarred Error2: 58975701049.680244
    Train Mean Square Root Error2 232791.2401396758
    Test Mean Square Root Error2 242849.13228109392


## Lastly V-Fold Cross Validation

Here v-fold cross validation was performed and then the mean squared error was calculated for both models. 


```python
#Cross_Val Model 1.3
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

cv_5_results1 = cross_val_score(linreg, X1, y1, cv=5, scoring="neg_mean_squared_error")
np.sqrt(-1*(cv_5_results1))
```




    array([2.62930429e+15, 2.30137459e+05, 2.19539322e+05, 2.23039033e+05,
           2.51333551e+05])




```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

cv_5_results2 = cross_val_score(linreg, X2, y2, cv=5, scoring="neg_mean_squared_error")
np.sqrt(-1*(cv_5_results2))
```




    array([1.53738082e+16, 2.35357359e+05, 2.25014163e+05, 2.28820582e+05,
           2.49074069e+05])


