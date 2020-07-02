```python
import pandas as pd
data = pd.read_csv(r"C:\Users\Sachin\Desktop\Housing_Data.csv")
```


```python

```


```python
data.dtypes
```




    price             float64
    lotsize             int64
    bedrooms            int64
    bathrooms           int64
    driveway            int32
    recroom             int32
    fullbase            int32
    gashw               int32
    airconditional      int32
    garage              int64
    prefarea            int32
    stories_four        uint8
    stories_one         uint8
    stories_three       uint8
    stories_two         uint8
    dtype: object




```python
# Preprocessing the data using LabelBinarizer() and OneHotEncoding Technique get_dummies() methods

import sklearn.preprocessing as pp
lb = pp.LabelBinarizer()
data.driveway = lb.fit_transform(data.driveway)
data.recroom = lb.fit_transform(data.recroom)
data.fullbase = lb.fit_transform(data.fullbase)
data.gashw = lb.fit_transform(data.gashw)
data.airconditional = lb.fit_transform(data.airconditional)
data.prefarea = lb.fit_transform(data.prefarea)
```


```python
data = pd.get_dummies(data , prefix="stories" , columns=["stories"])
```


```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>lotsize</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>driveway</th>
      <th>recroom</th>
      <th>fullbase</th>
      <th>gashw</th>
      <th>airconditional</th>
      <th>garage</th>
      <th>prefarea</th>
      <th>stories_four</th>
      <th>stories_one</th>
      <th>stories_three</th>
      <th>stories_two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42000.0</td>
      <td>5850</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38500.0</td>
      <td>4000</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49500.0</td>
      <td>3060</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60500.0</td>
      <td>6650</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61000.0</td>
      <td>6360</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>66000.0</td>
      <td>4160</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>66000.0</td>
      <td>3880</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>69000.0</td>
      <td>4160</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>83800.0</td>
      <td>4800</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>88500.0</td>
      <td>5500</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# storing predicted value and influenced value

dependent = data.iloc[: , 0].values
independent = data.iloc[: , 1:15].values
```


```python
# Normalize the data into 0 - 1 form by using min-max scalling feature

from sklearn.preprocessing import MinMaxScaler
independent_norm_value = MinMaxScaler().fit_transform(independent)
```


```python
independent_norm_value
```




    array([[0.28865979, 0.4       , 0.        , ..., 0.        , 0.        ,
            1.        ],
           [0.16151203, 0.2       , 0.        , ..., 1.        , 0.        ,
            0.        ],
           [0.09690722, 0.4       , 0.        , ..., 1.        , 0.        ,
            0.        ],
           ...,
           [0.29896907, 0.4       , 0.33333333, ..., 0.        , 0.        ,
            0.        ],
           [0.29896907, 0.4       , 0.33333333, ..., 0.        , 0.        ,
            1.        ],
           [0.29896907, 0.4       , 0.        , ..., 0.        , 0.        ,
            1.        ]])




```python
data.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>lotsize</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>driveway</th>
      <th>recroom</th>
      <th>fullbase</th>
      <th>gashw</th>
      <th>airconditional</th>
      <th>garage</th>
      <th>prefarea</th>
      <th>stories_four</th>
      <th>stories_one</th>
      <th>stories_three</th>
      <th>stories_two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>price</th>
      <td>1.000000</td>
      <td>0.535796</td>
      <td>0.366447</td>
      <td>0.516719</td>
      <td>0.297167</td>
      <td>0.254960</td>
      <td>0.186218</td>
      <td>0.092837</td>
      <td>0.453347</td>
      <td>0.383302</td>
      <td>0.329074</td>
      <td>0.372281</td>
      <td>-0.270058</td>
      <td>0.138254</td>
      <td>-0.002089</td>
    </tr>
    <tr>
      <th>lotsize</th>
      <td>0.535796</td>
      <td>1.000000</td>
      <td>0.151851</td>
      <td>0.193833</td>
      <td>0.288778</td>
      <td>0.140327</td>
      <td>0.047487</td>
      <td>-0.009201</td>
      <td>0.221765</td>
      <td>0.352872</td>
      <td>0.234782</td>
      <td>0.178354</td>
      <td>0.054348</td>
      <td>0.020567</td>
      <td>-0.159612</td>
    </tr>
    <tr>
      <th>bedrooms</th>
      <td>0.366447</td>
      <td>0.151851</td>
      <td>1.000000</td>
      <td>0.373769</td>
      <td>-0.011996</td>
      <td>0.080492</td>
      <td>0.097201</td>
      <td>0.046028</td>
      <td>0.160412</td>
      <td>0.139117</td>
      <td>0.078953</td>
      <td>0.145525</td>
      <td>-0.509974</td>
      <td>0.099150</td>
      <td>0.377424</td>
    </tr>
    <tr>
      <th>bathrooms</th>
      <td>0.516719</td>
      <td>0.193833</td>
      <td>0.373769</td>
      <td>1.000000</td>
      <td>0.041955</td>
      <td>0.126892</td>
      <td>0.102791</td>
      <td>0.067365</td>
      <td>0.184955</td>
      <td>0.178178</td>
      <td>0.064013</td>
      <td>0.281003</td>
      <td>-0.250784</td>
      <td>0.036027</td>
      <td>0.080982</td>
    </tr>
    <tr>
      <th>driveway</th>
      <td>0.297167</td>
      <td>0.288778</td>
      <td>-0.011996</td>
      <td>0.041955</td>
      <td>1.000000</td>
      <td>0.091959</td>
      <td>0.043428</td>
      <td>-0.011942</td>
      <td>0.106290</td>
      <td>0.203682</td>
      <td>0.199378</td>
      <td>0.115453</td>
      <td>-0.053249</td>
      <td>0.073533</td>
      <td>-0.047074</td>
    </tr>
    <tr>
      <th>recroom</th>
      <td>0.254960</td>
      <td>0.140327</td>
      <td>0.080492</td>
      <td>0.126892</td>
      <td>0.091959</td>
      <td>1.000000</td>
      <td>0.372434</td>
      <td>-0.010119</td>
      <td>0.136626</td>
      <td>0.038122</td>
      <td>0.161292</td>
      <td>0.067567</td>
      <td>-0.022632</td>
      <td>-0.038733</td>
      <td>0.006938</td>
    </tr>
    <tr>
      <th>fullbase</th>
      <td>0.186218</td>
      <td>0.047487</td>
      <td>0.097201</td>
      <td>0.102791</td>
      <td>0.043428</td>
      <td>0.372434</td>
      <td>1.000000</td>
      <td>0.004677</td>
      <td>0.045248</td>
      <td>0.052524</td>
      <td>0.228651</td>
      <td>-0.165285</td>
      <td>0.059154</td>
      <td>-0.132540</td>
      <td>0.098694</td>
    </tr>
    <tr>
      <th>gashw</th>
      <td>0.092837</td>
      <td>-0.009201</td>
      <td>0.046028</td>
      <td>0.067365</td>
      <td>-0.011942</td>
      <td>-0.010119</td>
      <td>0.004677</td>
      <td>1.000000</td>
      <td>-0.130350</td>
      <td>0.068144</td>
      <td>-0.059170</td>
      <td>-0.062416</td>
      <td>-0.060336</td>
      <td>0.072922</td>
      <td>0.054823</td>
    </tr>
    <tr>
      <th>airconditional</th>
      <td>0.453347</td>
      <td>0.221765</td>
      <td>0.160412</td>
      <td>0.184955</td>
      <td>0.106290</td>
      <td>0.136626</td>
      <td>0.045248</td>
      <td>-0.130350</td>
      <td>1.000000</td>
      <td>0.156596</td>
      <td>0.115626</td>
      <td>0.298887</td>
      <td>-0.143174</td>
      <td>0.110682</td>
      <td>-0.074706</td>
    </tr>
    <tr>
      <th>garage</th>
      <td>0.383302</td>
      <td>0.352872</td>
      <td>0.139117</td>
      <td>0.178178</td>
      <td>0.203682</td>
      <td>0.038122</td>
      <td>0.052524</td>
      <td>0.068144</td>
      <td>0.156596</td>
      <td>1.000000</td>
      <td>0.092364</td>
      <td>0.126112</td>
      <td>0.016610</td>
      <td>-0.079170</td>
      <td>-0.041931</td>
    </tr>
    <tr>
      <th>prefarea</th>
      <td>0.329074</td>
      <td>0.234782</td>
      <td>0.078953</td>
      <td>0.064013</td>
      <td>0.199378</td>
      <td>0.161292</td>
      <td>0.228651</td>
      <td>-0.059170</td>
      <td>0.115626</td>
      <td>0.092364</td>
      <td>1.000000</td>
      <td>-0.010035</td>
      <td>-0.010668</td>
      <td>0.143067</td>
      <td>-0.059240</td>
    </tr>
    <tr>
      <th>stories_four</th>
      <td>0.372281</td>
      <td>0.178354</td>
      <td>0.145525</td>
      <td>0.281003</td>
      <td>0.115453</td>
      <td>0.067567</td>
      <td>-0.165285</td>
      <td>-0.062416</td>
      <td>0.298887</td>
      <td>0.126112</td>
      <td>-0.010035</td>
      <td>1.000000</td>
      <td>-0.240361</td>
      <td>-0.080113</td>
      <td>-0.250472</td>
    </tr>
    <tr>
      <th>stories_one</th>
      <td>-0.270058</td>
      <td>0.054348</td>
      <td>-0.509974</td>
      <td>-0.250784</td>
      <td>-0.053249</td>
      <td>-0.022632</td>
      <td>0.059154</td>
      <td>-0.060336</td>
      <td>-0.143174</td>
      <td>0.016610</td>
      <td>-0.010668</td>
      <td>-0.240361</td>
      <td>1.000000</td>
      <td>-0.237177</td>
      <td>-0.741533</td>
    </tr>
    <tr>
      <th>stories_three</th>
      <td>0.138254</td>
      <td>0.020567</td>
      <td>0.099150</td>
      <td>0.036027</td>
      <td>0.073533</td>
      <td>-0.038733</td>
      <td>-0.132540</td>
      <td>0.072922</td>
      <td>0.110682</td>
      <td>-0.079170</td>
      <td>0.143067</td>
      <td>-0.080113</td>
      <td>-0.237177</td>
      <td>1.000000</td>
      <td>-0.247154</td>
    </tr>
    <tr>
      <th>stories_two</th>
      <td>-0.002089</td>
      <td>-0.159612</td>
      <td>0.377424</td>
      <td>0.080982</td>
      <td>-0.047074</td>
      <td>0.006938</td>
      <td>0.098694</td>
      <td>0.054823</td>
      <td>-0.074706</td>
      <td>-0.041931</td>
      <td>-0.059240</td>
      <td>-0.250472</td>
      <td>-0.741533</td>
      <td>-0.247154</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# train the model using Linear Regression

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(independent , dependent)
```




    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)




```python
dependent_predict = lr.predict(independent)
```


```python
data.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>lotsize</th>
      <th>bedrooms</th>
      <th>bathrooms</th>
      <th>driveway</th>
      <th>recroom</th>
      <th>fullbase</th>
      <th>gashw</th>
      <th>airconditional</th>
      <th>garage</th>
      <th>prefarea</th>
      <th>stories_four</th>
      <th>stories_one</th>
      <th>stories_three</th>
      <th>stories_two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>42000.0</td>
      <td>5850</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>38500.0</td>
      <td>4000</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49500.0</td>
      <td>3060</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>60500.0</td>
      <td>6650</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>61000.0</td>
      <td>6360</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>66000.0</td>
      <td>4160</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>66000.0</td>
      <td>3880</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>69000.0</td>
      <td>4160</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>83800.0</td>
      <td>4800</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>88500.0</td>
      <td>5500</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Now we have to perform R-Squared for good fit model

# R-squared = total sum of squared residual (SSR) / sum of squared total (SST)

# SSR = (ith value of predicted Variable - mean of dependent variable)
# SST = (ith value of dependent column - mean of dependent variable)

import numpy as np
from sklearn.metrics import r2_score , mean_squared_error
data_accuracy = r2_score(dependent , dependent_predict)
print("data accuracy using R-squared ", data_accuracy)
rmse = np.sqrt(mean_squared_error(dependent , dependent_predict))
print("Root Mean Squared Error (RMSE)", rmse)
```

    data accuracy using R-squared  0.6736291293426986
    Root Mean Squared Error (RMSE) 15240.960183553374
    


```python
# Oridinary Least Squared model (OLS)

import statsmodels.api as sm
model = sm.OLS(dependent , independent)
model = model.fit()
model.summary()

    # Right now the accuracy is 0.674 that mean 67.4%
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th> <td>   0.674</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.666</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   84.47</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 02 Jul 2020</td> <th>  Prob (F-statistic):</th> <td>4.12e-120</td>
</tr>
<tr>
  <th>Time:</th>                 <td>13:19:44</td>     <th>  Log-Likelihood:    </th> <td> -6033.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   546</td>      <th>  AIC:               </th> <td>1.210e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   532</td>      <th>  BIC:               </th> <td>1.216e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    13</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>      <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th>  <td>    3.4853</td> <td>    0.357</td> <td>    9.760</td> <td> 0.000</td> <td>    2.784</td> <td>    4.187</td>
</tr>
<tr>
  <th>x2</th>  <td> 2207.4007</td> <td> 1126.826</td> <td>    1.959</td> <td> 0.051</td> <td>   -6.174</td> <td> 4420.976</td>
</tr>
<tr>
  <th>x3</th>  <td> 1.423e+04</td> <td> 1501.004</td> <td>    9.479</td> <td> 0.000</td> <td> 1.13e+04</td> <td> 1.72e+04</td>
</tr>
<tr>
  <th>x4</th>  <td> 6744.5906</td> <td> 2049.077</td> <td>    3.292</td> <td> 0.001</td> <td> 2719.317</td> <td> 1.08e+04</td>
</tr>
<tr>
  <th>x5</th>  <td> 4452.7280</td> <td> 1905.476</td> <td>    2.337</td> <td> 0.020</td> <td>  709.547</td> <td> 8195.909</td>
</tr>
<tr>
  <th>x6</th>  <td> 5611.2079</td> <td> 1602.026</td> <td>    3.503</td> <td> 0.000</td> <td> 2464.134</td> <td> 8758.281</td>
</tr>
<tr>
  <th>x7</th>  <td> 1.298e+04</td> <td> 3244.074</td> <td>    4.002</td> <td> 0.000</td> <td> 6610.754</td> <td> 1.94e+04</td>
</tr>
<tr>
  <th>x8</th>  <td> 1.246e+04</td> <td> 1568.474</td> <td>    7.944</td> <td> 0.000</td> <td> 9379.277</td> <td> 1.55e+04</td>
</tr>
<tr>
  <th>x9</th>  <td> 4207.8472</td> <td>  847.752</td> <td>    4.964</td> <td> 0.000</td> <td> 2542.495</td> <td> 5873.200</td>
</tr>
<tr>
  <th>x10</th> <td> 9339.4245</td> <td> 1695.216</td> <td>    5.509</td> <td> 0.000</td> <td> 6009.286</td> <td> 1.27e+04</td>
</tr>
<tr>
  <th>x11</th> <td> 2.265e+04</td> <td> 5140.674</td> <td>    4.406</td> <td> 0.000</td> <td> 1.25e+04</td> <td> 3.27e+04</td>
</tr>
<tr>
  <th>x12</th> <td> 2381.0771</td> <td> 3517.544</td> <td>    0.677</td> <td> 0.499</td> <td>-4528.903</td> <td> 9291.057</td>
</tr>
<tr>
  <th>x13</th> <td> 1.512e+04</td> <td> 4848.135</td> <td>    3.119</td> <td> 0.002</td> <td> 5596.507</td> <td> 2.46e+04</td>
</tr>
<tr>
  <th>x14</th> <td> 7666.2541</td> <td> 4153.290</td> <td>    1.846</td> <td> 0.065</td> <td> -492.606</td> <td> 1.58e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>98.599</td> <th>  Durbin-Watson:     </th> <td>   1.603</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 266.624</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.891</td> <th>  Prob(JB):          </th> <td>1.27e-58</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.923</td> <th>  Cond. No.          </th> <td>7.07e+04</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 7.07e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
# Remove multicollinearity and Calculate Variance influence factor

from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
inde_var = data.columns
inde_var = inde_var.delete(0)

for i in range(len(inde_var)):
    vif_list = [vif(independent , index) for index in range(len(inde_var))]
    max_vif = max(vif_list)
    print("Max VIF Value is" , max_vif)
    drop = vif_list.index(max_vif)
    print("For the independent variable:- " , inde_var[drop])
    if max_vif > 5:
            print("Deleting" , inde_var[drop])
            inde_var = inde_var.delete(drop)
            
print("Final independent Variable:- " , inde_var)
```

    Max VIF Value is 17.220894903652624
    For the independent variable:-  stories_two
    Deleting stories_two
    Max VIF Value is 11.781465345739338
    For the independent variable:-  stories_one
    Deleting stories_one
    Max VIF Value is 11.781465345739338
    For the independent variable:-  stories_three
    Deleting stories_three
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Max VIF Value is 4.544836339845972
    For the independent variable:-  stories_four
    Final independent Variable:-  Index(['lotsize', 'bedrooms', 'bathrooms', 'driveway', 'recroom', 'fullbase',
           'gashw', 'airconditional', 'garage', 'prefarea', 'stories_four'],
          dtype='object')
    


```python
# Check again the data accuracy using Ordinary Least Squared Model (OLS)

new_independent = data[inde_var].values
new_dependent = data['price'].values
model = sm.OLS(new_dependent , new_independent)
model = model.fit()
model.summary()

    # now after removing multicollinearity the accuracy of model is 0.955 that means 95.5%
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared (uncentered):</th>      <td>   0.955</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared (uncentered):</th> <td>   0.954</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>          <td>   1028.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Thu, 02 Jul 2020</td> <th>  Prob (F-statistic):</th>           <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>13:19:44</td>     <th>  Log-Likelihood:    </th>          <td> -6044.6</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   546</td>      <th>  AIC:               </th>          <td>1.211e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   535</td>      <th>  BIC:               </th>          <td>1.216e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    11</td>      <th>                     </th>              <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>              <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
   <td></td>      <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>x1</th>  <td>    3.2452</td> <td>    0.345</td> <td>    9.413</td> <td> 0.000</td> <td>    2.568</td> <td>    3.922</td>
</tr>
<tr>
  <th>x2</th>  <td> 4205.7971</td> <td>  765.362</td> <td>    5.495</td> <td> 0.000</td> <td> 2702.314</td> <td> 5709.280</td>
</tr>
<tr>
  <th>x3</th>  <td> 1.467e+04</td> <td> 1486.437</td> <td>    9.871</td> <td> 0.000</td> <td> 1.18e+04</td> <td> 1.76e+04</td>
</tr>
<tr>
  <th>x4</th>  <td> 7908.4756</td> <td> 1859.409</td> <td>    4.253</td> <td> 0.000</td> <td> 4255.838</td> <td> 1.16e+04</td>
</tr>
<tr>
  <th>x5</th>  <td> 4462.7103</td> <td> 1935.810</td> <td>    2.305</td> <td> 0.022</td> <td>  659.991</td> <td> 8265.430</td>
</tr>
<tr>
  <th>x6</th>  <td> 4247.1213</td> <td> 1594.294</td> <td>    2.664</td> <td> 0.008</td> <td> 1115.277</td> <td> 7378.965</td>
</tr>
<tr>
  <th>x7</th>  <td> 1.458e+04</td> <td> 3279.922</td> <td>    4.444</td> <td> 0.000</td> <td> 8133.296</td> <td>  2.1e+04</td>
</tr>
<tr>
  <th>x8</th>  <td> 1.343e+04</td> <td> 1575.185</td> <td>    8.529</td> <td> 0.000</td> <td> 1.03e+04</td> <td> 1.65e+04</td>
</tr>
<tr>
  <th>x9</th>  <td> 3681.4212</td> <td>  846.491</td> <td>    4.349</td> <td> 0.000</td> <td> 2018.568</td> <td> 5344.274</td>
</tr>
<tr>
  <th>x10</th> <td> 1.021e+04</td> <td> 1692.052</td> <td>    6.033</td> <td> 0.000</td> <td> 6884.316</td> <td> 1.35e+04</td>
</tr>
<tr>
  <th>x11</th> <td> 1.526e+04</td> <td> 2822.680</td> <td>    5.408</td> <td> 0.000</td> <td> 9719.938</td> <td> 2.08e+04</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>108.679</td> <th>  Durbin-Watson:     </th> <td>   1.595</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td> 298.470</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.974</td>  <th>  Prob(JB):          </th> <td>1.54e-65</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 6.054</td>  <th>  Cond. No.          </th> <td>2.75e+04</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 2.75e+04. This might indicate that there are<br/>strong multicollinearity or other numerical problems.




```python
user_input = {}
for var in inde_var:
    temp = input("Enter " +var+ " :- ")  
    user_input[var] = temp
user_df = pd.DataFrame(data=user_input , index=[0], columns=inde_var)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(new_independent , new_dependent)
price = lr.predict(user_df)
print("House Price in USD are :- " , int(price[0]))

# driveway , recroom , fullbase , gashw , airconditional , prefarea
# for these column above (0 = no , 1 = yes)
```

    Enter lotsize :- 2000
    Enter bedrooms :- 3
    Enter bathrooms :- 2
    Enter driveway :- 0
    Enter recroom :- 1
    Enter fullbase :- 0
    Enter gashw :- 1
    Enter airconditional :- 1
    Enter garage :- 2
    Enter prefarea :- 1
    Enter stories_four :- 1
    House Price in USD are :-  113595
    


```python

```


```python

```
