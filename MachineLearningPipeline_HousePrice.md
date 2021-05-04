## Machine Learning Pipelines: House Prices
 #### Problem Statement:   
 To predict the house price based on various features.     
 https://www.kaggle.com/c/house-prices-advanced-regression-techniques  
      
      
 #### DateSet:   
 https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data  
     

### LifeCycle In A Datascience Project  
1. Data Analysis  
2. Feature Engineering  
3. Feature Selection  
4. Model Building  
5. MOdel Deployment  

### Data Analysis  



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

##Display all the columns of the dataframe
pd.pandas.set_option('display.max_columns', None)
```


```python
dataset = pd.read_csv('C:/Users/dhany/Desktop/Housing_data/house-prices-advanced-regression-techniques/train.csv')
print(dataset.shape)
```

    (1460, 81)
    


```python
dataset.head()
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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>None</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>



#### Analyzing the following factors:  
1. Missing Values  
2. All the Numerical Variables  
3. Distribution of the Numerical Variables  
4. Categorical Variables  
5. Cardinality of Categorical Variables  
6. Outliers  
7. Relationship between the independent and dependent feature(Sales Price)  

### Missing Values:  
Here we will check how many missing values(percentage) are present in each feature .   
step 1: Make the list of features which has the missing values.    
step2: Print the feature name and the percentage of missing values.  


```python
features_na = [features for features in dataset.columns if dataset[features].isnull().sum()>1]
for feature in features_na:
    print(feature, np.round(dataset[feature].isnull().mean(),4), '%missing values')
```

    LotFrontage 0.1774 %missing values
    Alley 0.9377 %missing values
    MasVnrType 0.0055 %missing values
    MasVnrArea 0.0055 %missing values
    BsmtQual 0.0253 %missing values
    BsmtCond 0.0253 %missing values
    BsmtExposure 0.026 %missing values
    BsmtFinType1 0.0253 %missing values
    BsmtFinType2 0.026 %missing values
    FireplaceQu 0.4726 %missing values
    GarageType 0.0555 %missing values
    GarageYrBlt 0.0555 %missing values
    GarageFinish 0.0555 %missing values
    GarageQual 0.0555 %missing values
    GarageCond 0.0555 %missing values
    PoolQC 0.9952 %missing values
    Fence 0.8075 %missing values
    MiscFeature 0.963 %missing values
    

To check the impact of missing value i.e. if the sales price is going up or decreasing depending on the missing value,
we will plot these count of missing values for each feature against the sales price.


```python
for feature in features_na:
    data = dataset.copy()

    # 1 indicates Missing value is present and zero indicates no Missing value
    data[feature] = np.where(data[feature].isnull(),1,0)
    
    #let's calculate the mean saleprice where the information is missing or present
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()
```


    
![png](output_11_0.png)
    



    
![png](output_11_1.png)
    



    
![png](output_11_2.png)
    



    
![png](output_11_3.png)
    



    
![png](output_11_4.png)
    



    
![png](output_11_5.png)
    



    
![png](output_11_6.png)
    



    
![png](output_11_7.png)
    



    
![png](output_11_8.png)
    



    
![png](output_11_9.png)
    



    
![png](output_11_10.png)
    



    
![png](output_11_11.png)
    



    
![png](output_11_12.png)
    



    
![png](output_11_13.png)
    



    
![png](output_11_14.png)
    



    
![png](output_11_15.png)
    



    
![png](output_11_16.png)
    



    
![png](output_11_17.png)
    


In LotFrontage and other features, with the NaN value, the house price is showing high mean Sales price. So we need to replace these missing values and this will be done in the feature Engineering part.

In the dataset there is a feature called id, which is unique. So we dont need this feature.  
Lets print and see the number of ids.  


```python
print("Id:{}".format(len(dataset.Id)))
```

    Id:1460
    

### Numerical Variables  
The numerical variables or features in the dataset is determined below.  


```python
#The O represents an object variable. Determining the variables which are not Object is done below.
numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
print(len(numerical_features))
dataset[numerical_features].head()
```

    38
    




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
      <th>Id</th>
      <th>MSSubClass</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>MasVnrArea</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageYrBlt</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>65.0</td>
      <td>8450</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>196.0</td>
      <td>706</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2003.0</td>
      <td>2</td>
      <td>548</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>80.0</td>
      <td>9600</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>0.0</td>
      <td>978</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1976.0</td>
      <td>2</td>
      <td>460</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>68.0</td>
      <td>11250</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>162.0</td>
      <td>486</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2001.0</td>
      <td>2</td>
      <td>608</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>60.0</td>
      <td>9550</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>0.0</td>
      <td>216</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1998.0</td>
      <td>3</td>
      <td>642</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>84.0</td>
      <td>14260</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>350.0</td>
      <td>655</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>2000.0</td>
      <td>3</td>
      <td>836</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>



### Temporal Variables(Date Time Variables)  
From the dataset we have 4 year variables. We can extract information number of years or number of days. Information like the year house was built and the year house was sold is one such important information.  


```python
year_feature = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature ]
year_feature
```




    ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']




```python
for feature in year_feature:
    print (feature, dataset[feature].unique())
```

    YearBuilt [2003 1976 2001 1915 2000 1993 2004 1973 1931 1939 1965 2005 1962 2006
     1960 1929 1970 1967 1958 1930 2002 1968 2007 1951 1957 1927 1920 1966
     1959 1994 1954 1953 1955 1983 1975 1997 1934 1963 1981 1964 1999 1972
     1921 1945 1982 1998 1956 1948 1910 1995 1991 2009 1950 1961 1977 1985
     1979 1885 1919 1990 1969 1935 1988 1971 1952 1936 1923 1924 1984 1926
     1940 1941 1987 1986 2008 1908 1892 1916 1932 1918 1912 1947 1925 1900
     1980 1989 1992 1949 1880 1928 1978 1922 1996 2010 1946 1913 1937 1942
     1938 1974 1893 1914 1906 1890 1898 1904 1882 1875 1911 1917 1872 1905]
    YearRemodAdd [2003 1976 2002 1970 2000 1995 2005 1973 1950 1965 2006 1962 2007 1960
     2001 1967 2004 2008 1997 1959 1990 1955 1983 1980 1966 1963 1987 1964
     1972 1996 1998 1989 1953 1956 1968 1981 1992 2009 1982 1961 1993 1999
     1985 1979 1977 1969 1958 1991 1971 1952 1975 2010 1984 1986 1994 1988
     1954 1957 1951 1978 1974]
    GarageYrBlt [2003. 1976. 2001. 1998. 2000. 1993. 2004. 1973. 1931. 1939. 1965. 2005.
     1962. 2006. 1960. 1991. 1970. 1967. 1958. 1930. 2002. 1968. 2007. 2008.
     1957. 1920. 1966. 1959. 1995. 1954. 1953.   nan 1983. 1977. 1997. 1985.
     1963. 1981. 1964. 1999. 1935. 1990. 1945. 1987. 1989. 1915. 1956. 1948.
     1974. 2009. 1950. 1961. 1921. 1900. 1979. 1951. 1969. 1936. 1975. 1971.
     1923. 1984. 1926. 1955. 1986. 1988. 1916. 1932. 1972. 1918. 1980. 1924.
     1996. 1940. 1949. 1994. 1910. 1978. 1982. 1992. 1925. 1941. 2010. 1927.
     1947. 1937. 1942. 1938. 1952. 1928. 1922. 1934. 1906. 1914. 1946. 1908.
     1929. 1933.]
    YrSold [2008 2007 2006 2009 2010]
    


```python
#Let us analyze Temporal Date Time variables.
#checking relation between Year built and Year sold.

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Meadian House Price')
plt.title('House Price vs YearSold')
```




    Text(0.5, 1.0, 'House Price vs YearSold')




    
![png](output_20_1.png)
    


Here the price is seen decreasing. So we will find the problem with yearsold. We will compare the yearsold with other year features in the dataset.


```python
for feature in year_feature:
    if feature!= 'YrSold':
        data = dataset.copy()
        ## we will capture the difference between the year variable and year the house 
        data[feature] = data['YrSold']-data[feature]
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.show()
```


    
![png](output_22_0.png)
    



    
![png](output_22_1.png)
    



    
![png](output_22_2.png)
    


### Type of Numerical Variable- Discrete or Continuous  
The numerical variables are of two types. The discrete variables are those which has more than a threshold amount of unique values. Other wise it is a continuous variable. The continuous variable has other properties also like they should not be a part of year variables or id.

#### Discrete Variable


```python
discrete_features = [feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]
print(len(discrete_features))
```

    17
    


```python
discrete_features
```




    ['MSSubClass',
     'OverallQual',
     'OverallCond',
     'LowQualFinSF',
     'BsmtFullBath',
     'BsmtHalfBath',
     'FullBath',
     'HalfBath',
     'BedroomAbvGr',
     'KitchenAbvGr',
     'TotRmsAbvGrd',
     'Fireplaces',
     'GarageCars',
     '3SsnPorch',
     'PoolArea',
     'MiscVal',
     'MoSold']




```python
dataset[discrete_features].head()
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
      <th>MSSubClass</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>LowQualFinSF</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>TotRmsAbvGrd</th>
      <th>Fireplaces</th>
      <th>GarageCars</th>
      <th>3SsnPorch</th>
      <th>PoolArea</th>
      <th>MiscVal</th>
      <th>MoSold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>60</td>
      <td>7</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>20</td>
      <td>6</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>60</td>
      <td>7</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>70</td>
      <td>7</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>60</td>
      <td>8</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



### RelationShip Between discrete values and SalesPrice


```python
for feature in discrete_features:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
```


    
![png](output_29_0.png)
    



    
![png](output_29_1.png)
    



    
![png](output_29_2.png)
    



    
![png](output_29_3.png)
    



    
![png](output_29_4.png)
    



    
![png](output_29_5.png)
    



    
![png](output_29_6.png)
    



    
![png](output_29_7.png)
    



    
![png](output_29_8.png)
    



    
![png](output_29_9.png)
    



    
![png](output_29_10.png)
    



    
![png](output_29_11.png)
    



    
![png](output_29_12.png)
    



    
![png](output_29_13.png)
    



    
![png](output_29_14.png)
    



    
![png](output_29_15.png)
    



    
![png](output_29_16.png)
    


### Continuous Variable  
It shoud not be a part of discrete variables, year or id


```python
continuous_features = [feature for feature in numerical_features if feature not in discrete_features+year_feature+['Id']]
print(len(continuous_features))
```

    16
    


```python
##These are continous values. So we need to plot them in histograms
for feature in continuous_features:
    data = dataset.copy()
    data[feature].hist(bins = 25)
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.show()
```


    
![png](output_32_0.png)
    



    
![png](output_32_1.png)
    



    
![png](output_32_2.png)
    



    
![png](output_32_3.png)
    



    
![png](output_32_4.png)
    



    
![png](output_32_5.png)
    



    
![png](output_32_6.png)
    



    
![png](output_32_7.png)
    



    
![png](output_32_8.png)
    



    
![png](output_32_9.png)
    



    
![png](output_32_10.png)
    



    
![png](output_32_11.png)
    



    
![png](output_32_12.png)
    



    
![png](output_32_13.png)
    



    
![png](output_32_14.png)
    



    
![png](output_32_15.png)
    


Many features have skewed data when continuous variables are plotted.The logarithmic transformation will be performed on these features' data.We are going to apply log normal distribution.


```python
data['BsmtHalfBath'].unique()
```




    array([0, 1, 2], dtype=int64)




```python

for feature in continuous_features:
    data = dataset.copy()
#log 0 is undefined. It's not a real number, because you can never get zero by raising anything to the power of anything else. 
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])#since there is skewness in the data distribution
        data['SalePrice'] = np.log(data['SalePrice'])#datadistribution is skewed
        plt.scatter(data[feature], data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalesPrice')
        plt.title(feature)
        plt.show()
```


    
![png](output_35_0.png)
    



    
![png](output_35_1.png)
    



    
![png](output_35_2.png)
    



    
![png](output_35_3.png)
    



    
![png](output_35_4.png)
    


#### Outliers  
Next step is to find out the outliers present inthe data. They are extreme values, a very high or low value.


```python

for feature in continuous_features:
    data = dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature] = np.log(data[feature])
        data.boxplot(column=feature)
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
```


    
![png](output_37_0.png)
    



    
![png](output_37_1.png)
    



    
![png](output_37_2.png)
    



    
![png](output_37_3.png)
    



    
![png](output_37_4.png)
    


There are a lot of outliers. The boxplot works only for continuous variables.  

### Categorical Variables



```python
categorical_features =  [feature for feature in dataset.columns if data[feature].dtypes == 'O']
categorical_features
```




    ['MSZoning',
     'Street',
     'Alley',
     'LotShape',
     'LandContour',
     'Utilities',
     'LotConfig',
     'LandSlope',
     'Neighborhood',
     'Condition1',
     'Condition2',
     'BldgType',
     'HouseStyle',
     'RoofStyle',
     'RoofMatl',
     'Exterior1st',
     'Exterior2nd',
     'MasVnrType',
     'ExterQual',
     'ExterCond',
     'Foundation',
     'BsmtQual',
     'BsmtCond',
     'BsmtExposure',
     'BsmtFinType1',
     'BsmtFinType2',
     'Heating',
     'HeatingQC',
     'CentralAir',
     'Electrical',
     'KitchenQual',
     'Functional',
     'FireplaceQu',
     'GarageType',
     'GarageFinish',
     'GarageQual',
     'GarageCond',
     'PavedDrive',
     'PoolQC',
     'Fence',
     'MiscFeature',
     'SaleType',
     'SaleCondition']




```python
dataset['ExterQual'][dataset['ExterQual']=='FA'].head()
```




    Series([], Name: ExterQual, dtype: object)



The main thing we need to focus on categorical features is their cardinality ( "number of elements" ). There are a lot of categoricl features in this dataset.


```python
for feature in categorical_features:
    print('The feature: {}; The categories:{}; count:{}'.format(feature, dataset[feature].unique(), len(dataset[feature].unique())))
```

    The feature: MSZoning; The categories:['RL' 'RM' 'C (all)' 'FV' 'RH']; count:5
    The feature: Street; The categories:['Pave' 'Grvl']; count:2
    The feature: Alley; The categories:[nan 'Grvl' 'Pave']; count:3
    The feature: LotShape; The categories:['Reg' 'IR1' 'IR2' 'IR3']; count:4
    The feature: LandContour; The categories:['Lvl' 'Bnk' 'Low' 'HLS']; count:4
    The feature: Utilities; The categories:['AllPub' 'NoSeWa']; count:2
    The feature: LotConfig; The categories:['Inside' 'FR2' 'Corner' 'CulDSac' 'FR3']; count:5
    The feature: LandSlope; The categories:['Gtl' 'Mod' 'Sev']; count:3
    The feature: Neighborhood; The categories:['CollgCr' 'Veenker' 'Crawfor' 'NoRidge' 'Mitchel' 'Somerst' 'NWAmes'
     'OldTown' 'BrkSide' 'Sawyer' 'NridgHt' 'NAmes' 'SawyerW' 'IDOTRR'
     'MeadowV' 'Edwards' 'Timber' 'Gilbert' 'StoneBr' 'ClearCr' 'NPkVill'
     'Blmngtn' 'BrDale' 'SWISU' 'Blueste']; count:25
    The feature: Condition1; The categories:['Norm' 'Feedr' 'PosN' 'Artery' 'RRAe' 'RRNn' 'RRAn' 'PosA' 'RRNe']; count:9
    The feature: Condition2; The categories:['Norm' 'Artery' 'RRNn' 'Feedr' 'PosN' 'PosA' 'RRAn' 'RRAe']; count:8
    The feature: BldgType; The categories:['1Fam' '2fmCon' 'Duplex' 'TwnhsE' 'Twnhs']; count:5
    The feature: HouseStyle; The categories:['2Story' '1Story' '1.5Fin' '1.5Unf' 'SFoyer' 'SLvl' '2.5Unf' '2.5Fin']; count:8
    The feature: RoofStyle; The categories:['Gable' 'Hip' 'Gambrel' 'Mansard' 'Flat' 'Shed']; count:6
    The feature: RoofMatl; The categories:['CompShg' 'WdShngl' 'Metal' 'WdShake' 'Membran' 'Tar&Grv' 'Roll'
     'ClyTile']; count:8
    The feature: Exterior1st; The categories:['VinylSd' 'MetalSd' 'Wd Sdng' 'HdBoard' 'BrkFace' 'WdShing' 'CemntBd'
     'Plywood' 'AsbShng' 'Stucco' 'BrkComm' 'AsphShn' 'Stone' 'ImStucc'
     'CBlock']; count:15
    The feature: Exterior2nd; The categories:['VinylSd' 'MetalSd' 'Wd Shng' 'HdBoard' 'Plywood' 'Wd Sdng' 'CmentBd'
     'BrkFace' 'Stucco' 'AsbShng' 'Brk Cmn' 'ImStucc' 'AsphShn' 'Stone'
     'Other' 'CBlock']; count:16
    The feature: MasVnrType; The categories:['BrkFace' 'None' 'Stone' 'BrkCmn' nan]; count:5
    The feature: ExterQual; The categories:['Gd' 'TA' 'Ex' 'Fa']; count:4
    The feature: ExterCond; The categories:['TA' 'Gd' 'Fa' 'Po' 'Ex']; count:5
    The feature: Foundation; The categories:['PConc' 'CBlock' 'BrkTil' 'Wood' 'Slab' 'Stone']; count:6
    The feature: BsmtQual; The categories:['Gd' 'TA' 'Ex' nan 'Fa']; count:5
    The feature: BsmtCond; The categories:['TA' 'Gd' nan 'Fa' 'Po']; count:5
    The feature: BsmtExposure; The categories:['No' 'Gd' 'Mn' 'Av' nan]; count:5
    The feature: BsmtFinType1; The categories:['GLQ' 'ALQ' 'Unf' 'Rec' 'BLQ' nan 'LwQ']; count:7
    The feature: BsmtFinType2; The categories:['Unf' 'BLQ' nan 'ALQ' 'Rec' 'LwQ' 'GLQ']; count:7
    The feature: Heating; The categories:['GasA' 'GasW' 'Grav' 'Wall' 'OthW' 'Floor']; count:6
    The feature: HeatingQC; The categories:['Ex' 'Gd' 'TA' 'Fa' 'Po']; count:5
    The feature: CentralAir; The categories:['Y' 'N']; count:2
    The feature: Electrical; The categories:['SBrkr' 'FuseF' 'FuseA' 'FuseP' 'Mix' nan]; count:6
    The feature: KitchenQual; The categories:['Gd' 'TA' 'Ex' 'Fa']; count:4
    The feature: Functional; The categories:['Typ' 'Min1' 'Maj1' 'Min2' 'Mod' 'Maj2' 'Sev']; count:7
    The feature: FireplaceQu; The categories:[nan 'TA' 'Gd' 'Fa' 'Ex' 'Po']; count:6
    The feature: GarageType; The categories:['Attchd' 'Detchd' 'BuiltIn' 'CarPort' nan 'Basment' '2Types']; count:7
    The feature: GarageFinish; The categories:['RFn' 'Unf' 'Fin' nan]; count:4
    The feature: GarageQual; The categories:['TA' 'Fa' 'Gd' nan 'Ex' 'Po']; count:6
    The feature: GarageCond; The categories:['TA' 'Fa' nan 'Gd' 'Po' 'Ex']; count:6
    The feature: PavedDrive; The categories:['Y' 'N' 'P']; count:3
    The feature: PoolQC; The categories:[nan 'Ex' 'Fa' 'Gd']; count:4
    The feature: Fence; The categories:[nan 'MnPrv' 'GdWo' 'GdPrv' 'MnWw']; count:5
    The feature: MiscFeature; The categories:[nan 'Shed' 'Gar2' 'Othr' 'TenC']; count:5
    The feature: SaleType; The categories:['WD' 'New' 'COD' 'ConLD' 'ConLI' 'CWD' 'ConLw' 'Con' 'Oth']; count:9
    The feature: SaleCondition; The categories:['Normal' 'Abnorml' 'Partial' 'AdjLand' 'Alloca' 'Family']; count:6
    

There are some categorical variables which has 15 or more categories. We will have to look into these variables to see how to handle them. Rest of the variables can be processed with one hot encoding.  
Next we will find out the relationship between our target variable sales price and the categorical variables.  


```python
for feature in categorical_features:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()
```


    
![png](output_44_0.png)
    



    
![png](output_44_1.png)
    



    
![png](output_44_2.png)
    



    
![png](output_44_3.png)
    



    
![png](output_44_4.png)
    



    
![png](output_44_5.png)
    



    
![png](output_44_6.png)
    



    
![png](output_44_7.png)
    



    
![png](output_44_8.png)
    



    
![png](output_44_9.png)
    



    
![png](output_44_10.png)
    



    
![png](output_44_11.png)
    



    
![png](output_44_12.png)
    



    
![png](output_44_13.png)
    



    
![png](output_44_14.png)
    



    
![png](output_44_15.png)
    



    
![png](output_44_16.png)
    



    
![png](output_44_17.png)
    



    
![png](output_44_18.png)
    



    
![png](output_44_19.png)
    



    
![png](output_44_20.png)
    



    
![png](output_44_21.png)
    



    
![png](output_44_22.png)
    



    
![png](output_44_23.png)
    



    
![png](output_44_24.png)
    



    
![png](output_44_25.png)
    



    
![png](output_44_26.png)
    



    
![png](output_44_27.png)
    



    
![png](output_44_28.png)
    



    
![png](output_44_29.png)
    



    
![png](output_44_30.png)
    



    
![png](output_44_31.png)
    



    
![png](output_44_32.png)
    



    
![png](output_44_33.png)
    



    
![png](output_44_34.png)
    



    
![png](output_44_35.png)
    



    
![png](output_44_36.png)
    



    
![png](output_44_37.png)
    



    
![png](output_44_38.png)
    



    
![png](output_44_39.png)
    



    
![png](output_44_40.png)
    



    
![png](output_44_41.png)
    



    
![png](output_44_42.png)
    



```python

```
