# TRABAJO DE FIN DE MASTER

## Deteccion de Transacciones fraudulentas en Ethereum 

- Sergio Alejandro Marchena Gordillo
- smarchenago@alumni.unav.es

## Dataset


Este conjunto de datos contiene filas de transacciones conocidas como fraudulentas y válidas realizadas en Ethereum, un tipo de criptomoneda.

A continuación, se describe el contenido de las filas del conjunto de datos:

- Index: the index number of a row
- Address: the address of the ethereum account
- FLAG: whether the transaction is fraud or not
- Avg min between sent tnx: Average time between sent transactions for account in minutes
- Avg min between received tnx: Average time between received transactions for account in minutes
- Time Diff between first and_last (Mins): Time difference between the first and last transaction
- Sent_tnx: Total number of sent normal transactions
- Received_tnx: Total number of received normal transactions
- NumberofCreated_Contracts: Total Number of created contract transactions
- UniqueReceivedFrom_Addresses: Total Unique addresses from which account received transaction
- UniqueSentTo_Addresses20: Total Unique addresses from which account sent transactions
- MinValueReceived: Minimum value in Ether ever received
- MaxValueReceived: Maximum value in Ether ever received
- AvgValueReceived5Average value in Ether ever received
- MinValSent: Minimum value of Ether ever sent
- MaxValSent: Maximum value of Ether ever sent
- AvgValSent: Average value of Ether ever sent
- MinValueSentToContract: Minimum value of Ether sent to a contract
- MaxValueSentToContract: Maximum value of Ether sent to a contract
- AvgValueSentToContract: Average value of Ether sent to contracts
- TotalTransactions(IncludingTnxtoCreate_Contract): Total number of transactions
- TotalEtherSent:Total Ether sent for account address
- TotalEtherReceived: Total Ether received for account address
- TotalEtherSent_Contracts: Total Ether sent to Contract addresses
- TotalEtherBalance: Total Ether Balance following enacted transactions
- TotalERC20Tnxs: Total number of ERC20 token transfer transactions
- ERC20TotalEther_Received: Total ERC20 token received transactions in Ether
- ERC20TotalEther_Sent: Total ERC20token sent transactions in Ether
- ERC20TotalEtherSentContract: Total ERC20 token transfer to other contracts in Ether
- ERC20UniqSent_Addr: Number of ERC20 token transactions sent to Unique account addresses
- ERC20UniqRec_Addr: Number of ERC20 token transactions received from Unique addresses
- ERC20UniqRecContractAddr: Number of ERC20token transactions received from Unique contract addresses
- ERC20AvgTimeBetweenSent_Tnx: Average time between ERC20 token sent transactions in minutes
- ERC20AvgTimeBetweenRec_Tnx: Average time between ERC20 token received transactions in minutes
- ERC20AvgTimeBetweenContract_Tnx: Average time ERC20 token between sent token transactions
- ERC20MinVal_Rec: Minimum value in Ether received from ERC20 token transactions for account
- ERC20MaxVal_Rec: Maximum value in Ether received from ERC20 token transactions for account
- ERC20AvgVal_Rec: Average value in Ether received from ERC20 token transactions for account
- ERC20MinVal_Sent: Minimum value in Ether sent from ERC20 token transactions for account
- ERC20MaxVal_Sent: Maximum value in Ether sent from ERC20 token transactions for account
- ERC20AvgVal_Sent: Average value in Ether sent from ERC20 token transactions for account
- ERC20UniqSentTokenName: Number of Unique ERC20 tokens transferred
- RC20UniqRecTokenName: Number of Unique ERC20 tokens received
- ERC20MostSentTokenType: Most sent token for account via ERC20 transaction
- ERC20MostRecTokenType: Most received token for account via ERC20 transactions

## Librerias


```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, classification_report
#from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import pickle
```

## Carga y lectura de datos


```python
!kaggle datasets download -d rupakroy/ethereum-fraud-detection


# se tienen que extraer el .CSV del zip que descarga esta comando en esta ubicacion
```

    Warning: Looks like you're using an outdated API Version, please consider updating (server 1.6.17 / client 1.6.14)
    Dataset URL: https://www.kaggle.com/datasets/rupakroy/ethereum-fraud-detection
    License(s): CC0-1.0
    ethereum-fraud-detection.zip: Skipping, found more recently modified local copy (use --force to force download)



```python
df =  pd.read_csv('transaction_dataset.csv', index_col=0)

print(df.shape)
df.head()

```

    (9841, 50)





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
      <th>Index</th>
      <th>Address</th>
      <th>FLAG</th>
      <th>Avg min between sent tnx</th>
      <th>Avg min between received tnx</th>
      <th>Time Diff between first and last (Mins)</th>
      <th>Sent tnx</th>
      <th>Received Tnx</th>
      <th>Number of Created Contracts</th>
      <th>Unique Received From Addresses</th>
      <th>...</th>
      <th>ERC20 min val sent</th>
      <th>ERC20 max val sent</th>
      <th>ERC20 avg val sent</th>
      <th>ERC20 min val sent contract</th>
      <th>ERC20 max val sent contract</th>
      <th>ERC20 avg val sent contract</th>
      <th>ERC20 uniq sent token name</th>
      <th>ERC20 uniq rec token name</th>
      <th>ERC20 most sent token type</th>
      <th>ERC20_most_rec_token_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0x00009277775ac7d0d59eaad8fee3d10ac6c805e8</td>
      <td>0</td>
      <td>844.26</td>
      <td>1093.71</td>
      <td>704785.63</td>
      <td>721</td>
      <td>89</td>
      <td>0</td>
      <td>40</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.683100e+07</td>
      <td>271779.920000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39.0</td>
      <td>57.0</td>
      <td>Cofoundit</td>
      <td>Numeraire</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0x0002b44ddb1476db43c868bd494422ee4c136fed</td>
      <td>0</td>
      <td>12709.07</td>
      <td>2958.44</td>
      <td>1218216.73</td>
      <td>94</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>...</td>
      <td>2.260809</td>
      <td>2.260809e+00</td>
      <td>2.260809</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>Livepeer Token</td>
      <td>Livepeer Token</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0x0002bda54cb772d040f779e88eb453cac0daa244</td>
      <td>0</td>
      <td>246194.54</td>
      <td>2434.02</td>
      <td>516729.30</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>10</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>XENON</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0x00038e6ba2fd5c09aedb96697c8d7b8fa6632e5e</td>
      <td>0</td>
      <td>10219.60</td>
      <td>15785.09</td>
      <td>397555.90</td>
      <td>25</td>
      <td>9</td>
      <td>0</td>
      <td>7</td>
      <td>...</td>
      <td>100.000000</td>
      <td>9.029231e+03</td>
      <td>3804.076893</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>Raiden</td>
      <td>XENON</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0x00062d1dd1afb6fb02540ddad9cdebfe568e0d89</td>
      <td>0</td>
      <td>36.61</td>
      <td>10707.77</td>
      <td>382472.42</td>
      <td>4598</td>
      <td>20</td>
      <td>1</td>
      <td>7</td>
      <td>...</td>
      <td>0.000000</td>
      <td>4.500000e+04</td>
      <td>13726.659220</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>27.0</td>
      <td>StatusNetwork</td>
      <td>EOS</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```python
# Eliminamos las primeras dos columnas (Index, Adress) ya que no son relevantes en este contexto
df = df.iloc[:,2:]
df.head()
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
      <th>FLAG</th>
      <th>Avg min between sent tnx</th>
      <th>Avg min between received tnx</th>
      <th>Time Diff between first and last (Mins)</th>
      <th>Sent tnx</th>
      <th>Received Tnx</th>
      <th>Number of Created Contracts</th>
      <th>Unique Received From Addresses</th>
      <th>Unique Sent To Addresses</th>
      <th>min value received</th>
      <th>...</th>
      <th>ERC20 min val sent</th>
      <th>ERC20 max val sent</th>
      <th>ERC20 avg val sent</th>
      <th>ERC20 min val sent contract</th>
      <th>ERC20 max val sent contract</th>
      <th>ERC20 avg val sent contract</th>
      <th>ERC20 uniq sent token name</th>
      <th>ERC20 uniq rec token name</th>
      <th>ERC20 most sent token type</th>
      <th>ERC20_most_rec_token_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>844.26</td>
      <td>1093.71</td>
      <td>704785.63</td>
      <td>721</td>
      <td>89</td>
      <td>0</td>
      <td>40</td>
      <td>118</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>1.683100e+07</td>
      <td>271779.920000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39.0</td>
      <td>57.0</td>
      <td>Cofoundit</td>
      <td>Numeraire</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>12709.07</td>
      <td>2958.44</td>
      <td>1218216.73</td>
      <td>94</td>
      <td>8</td>
      <td>0</td>
      <td>5</td>
      <td>14</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.260809</td>
      <td>2.260809e+00</td>
      <td>2.260809</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>Livepeer Token</td>
      <td>Livepeer Token</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>246194.54</td>
      <td>2434.02</td>
      <td>516729.30</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>10</td>
      <td>2</td>
      <td>0.113119</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>XENON</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>10219.60</td>
      <td>15785.09</td>
      <td>397555.90</td>
      <td>25</td>
      <td>9</td>
      <td>0</td>
      <td>7</td>
      <td>13</td>
      <td>0.000000</td>
      <td>...</td>
      <td>100.000000</td>
      <td>9.029231e+03</td>
      <td>3804.076893</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>11.0</td>
      <td>Raiden</td>
      <td>XENON</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>36.61</td>
      <td>10707.77</td>
      <td>382472.42</td>
      <td>4598</td>
      <td>20</td>
      <td>1</td>
      <td>7</td>
      <td>19</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>4.500000e+04</td>
      <td>13726.659220</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>27.0</td>
      <td>StatusNetwork</td>
      <td>EOS</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>



## Analisis de Datos Exploratorio (EDA)  


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 9841 entries, 0 to 9840
    Data columns (total 48 columns):
     #   Column                                                Non-Null Count  Dtype  
    ---  ------                                                --------------  -----  
     0   FLAG                                                  9841 non-null   int64  
     1   Avg min between sent tnx                              9841 non-null   float64
     2   Avg min between received tnx                          9841 non-null   float64
     3   Time Diff between first and last (Mins)               9841 non-null   float64
     4   Sent tnx                                              9841 non-null   int64  
     5   Received Tnx                                          9841 non-null   int64  
     6   Number of Created Contracts                           9841 non-null   int64  
     7   Unique Received From Addresses                        9841 non-null   int64  
     8   Unique Sent To Addresses                              9841 non-null   int64  
     9   min value received                                    9841 non-null   float64
     10  max value received                                    9841 non-null   float64
     11  avg val received                                      9841 non-null   float64
     12  min val sent                                          9841 non-null   float64
     13  max val sent                                          9841 non-null   float64
     14  avg val sent                                          9841 non-null   float64
     15  min value sent to contract                            9841 non-null   float64
     16  max val sent to contract                              9841 non-null   float64
     17  avg value sent to contract                            9841 non-null   float64
     18  total transactions (including tnx to create contract  9841 non-null   int64  
     19  total Ether sent                                      9841 non-null   float64
     20  total ether received                                  9841 non-null   float64
     21  total ether sent contracts                            9841 non-null   float64
     22  total ether balance                                   9841 non-null   float64
     23   Total ERC20 tnxs                                     9012 non-null   float64
     24   ERC20 total Ether received                           9012 non-null   float64
     25   ERC20 total ether sent                               9012 non-null   float64
     26   ERC20 total Ether sent contract                      9012 non-null   float64
     27   ERC20 uniq sent addr                                 9012 non-null   float64
     28   ERC20 uniq rec addr                                  9012 non-null   float64
     29   ERC20 uniq sent addr.1                               9012 non-null   float64
     30   ERC20 uniq rec contract addr                         9012 non-null   float64
     31   ERC20 avg time between sent tnx                      9012 non-null   float64
     32   ERC20 avg time between rec tnx                       9012 non-null   float64
     33   ERC20 avg time between rec 2 tnx                     9012 non-null   float64
     34   ERC20 avg time between contract tnx                  9012 non-null   float64
     35   ERC20 min val rec                                    9012 non-null   float64
     36   ERC20 max val rec                                    9012 non-null   float64
     37   ERC20 avg val rec                                    9012 non-null   float64
     38   ERC20 min val sent                                   9012 non-null   float64
     39   ERC20 max val sent                                   9012 non-null   float64
     40   ERC20 avg val sent                                   9012 non-null   float64
     41   ERC20 min val sent contract                          9012 non-null   float64
     42   ERC20 max val sent contract                          9012 non-null   float64
     43   ERC20 avg val sent contract                          9012 non-null   float64
     44   ERC20 uniq sent token name                           9012 non-null   float64
     45   ERC20 uniq rec token name                            9012 non-null   float64
     46   ERC20 most sent token type                           7144 non-null   object 
     47   ERC20_most_rec_token_type                            8970 non-null   object 
    dtypes: float64(39), int64(7), object(2)
    memory usage: 3.7+ MB



```python
# Cambiamos las unicas 2 variables tipo 'object' a categorias para mejor manejo y analisis.

categories = df.select_dtypes('O').columns.astype('category')
df[categories]
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
      <th>ERC20 most sent token type</th>
      <th>ERC20_most_rec_token_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cofoundit</td>
      <td>Numeraire</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Livepeer Token</td>
      <td>Livepeer Token</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>XENON</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Raiden</td>
      <td>XENON</td>
    </tr>
    <tr>
      <th>4</th>
      <td>StatusNetwork</td>
      <td>EOS</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9836</th>
      <td></td>
      <td>GSENetwork</td>
    </tr>
    <tr>
      <th>9837</th>
      <td></td>
      <td>Blockwell say NOTSAFU</td>
    </tr>
    <tr>
      <th>9838</th>
      <td></td>
      <td>Free BOB Tokens - BobsRepair.com</td>
    </tr>
    <tr>
      <th>9839</th>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9840</th>
      <td></td>
      <td>INS Promo1</td>
    </tr>
  </tbody>
</table>
<p>9841 rows × 2 columns</p>
</div>



### Variables categroricas

#### Valores unicos


```python
for i in df[categories].columns:
    print(i, "----> ", len(df[i].value_counts()), ' valores unicos')
```

     ERC20 most sent token type ---->  304  valores unicos
     ERC20_most_rec_token_type ---->  466  valores unicos


### Variables numericas


```python
numericals = df.select_dtypes(include=['float','int']).columns
df[numericals].describe()
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
      <th>FLAG</th>
      <th>Avg min between sent tnx</th>
      <th>Avg min between received tnx</th>
      <th>Time Diff between first and last (Mins)</th>
      <th>Sent tnx</th>
      <th>Received Tnx</th>
      <th>Number of Created Contracts</th>
      <th>Unique Received From Addresses</th>
      <th>Unique Sent To Addresses</th>
      <th>min value received</th>
      <th>...</th>
      <th>ERC20 max val rec</th>
      <th>ERC20 avg val rec</th>
      <th>ERC20 min val sent</th>
      <th>ERC20 max val sent</th>
      <th>ERC20 avg val sent</th>
      <th>ERC20 min val sent contract</th>
      <th>ERC20 max val sent contract</th>
      <th>ERC20 avg val sent contract</th>
      <th>ERC20 uniq sent token name</th>
      <th>ERC20 uniq rec token name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9841.000000</td>
      <td>9841.000000</td>
      <td>9841.000000</td>
      <td>9.841000e+03</td>
      <td>9841.000000</td>
      <td>9841.000000</td>
      <td>9841.000000</td>
      <td>9841.000000</td>
      <td>9841.000000</td>
      <td>9841.000000</td>
      <td>...</td>
      <td>9.012000e+03</td>
      <td>9.012000e+03</td>
      <td>9.012000e+03</td>
      <td>9.012000e+03</td>
      <td>9.012000e+03</td>
      <td>9012.0</td>
      <td>9012.0</td>
      <td>9012.0</td>
      <td>9012.000000</td>
      <td>9012.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.221421</td>
      <td>5086.878721</td>
      <td>8004.851184</td>
      <td>2.183333e+05</td>
      <td>115.931714</td>
      <td>163.700945</td>
      <td>3.729702</td>
      <td>30.360939</td>
      <td>25.840159</td>
      <td>43.845153</td>
      <td>...</td>
      <td>1.252524e+08</td>
      <td>4.346203e+06</td>
      <td>1.174126e+04</td>
      <td>1.303594e+07</td>
      <td>6.318389e+06</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.384931</td>
      <td>4.826676</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.415224</td>
      <td>21486.549974</td>
      <td>23081.714801</td>
      <td>3.229379e+05</td>
      <td>757.226361</td>
      <td>940.836550</td>
      <td>141.445583</td>
      <td>298.621112</td>
      <td>263.820410</td>
      <td>325.929139</td>
      <td>...</td>
      <td>1.053741e+10</td>
      <td>2.141192e+08</td>
      <td>1.053567e+06</td>
      <td>1.179905e+09</td>
      <td>5.914764e+08</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.735121</td>
      <td>16.678607</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.169300e+02</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.001000</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>17.340000</td>
      <td>509.770000</td>
      <td>4.663703e+04</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>0.095856</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000</td>
      <td>565.470000</td>
      <td>5480.390000</td>
      <td>3.040710e+05</td>
      <td>11.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>...</td>
      <td>9.900000e+01</td>
      <td>2.946467e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>430287.670000</td>
      <td>482175.490000</td>
      <td>1.954861e+06</td>
      <td>10000.000000</td>
      <td>10000.000000</td>
      <td>9995.000000</td>
      <td>9999.000000</td>
      <td>9287.000000</td>
      <td>10000.000000</td>
      <td>...</td>
      <td>1.000000e+12</td>
      <td>1.724181e+10</td>
      <td>1.000000e+08</td>
      <td>1.120000e+11</td>
      <td>5.614756e+10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>213.000000</td>
      <td>737.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 46 columns</p>
</div>



#### Varianza


```python
print("Varianza ordenada en forma descendente: ")
df[numericals].var().sort_values(ascending=False)
```

    Varianza ordenada en forma descendente: 





     ERC20 total Ether received                             1.110618e+20
     ERC20 max val rec                                      1.110370e+20
     ERC20 total ether sent                                 1.393321e+18
     ERC20 max val sent                                     1.392176e+18
     ERC20 avg val sent                                     3.498443e+17
     ERC20 avg val rec                                      4.584705e+16
     ERC20 min val sent                                     1.110004e+12
    total ether received                                    1.326451e+11
    total Ether sent                                        1.283952e+11
    Time Diff between first and last (Mins)                 1.042889e+11
    total ether balance                                     5.877009e+10
    Avg min between received tnx                            5.327656e+08
    Avg min between sent tnx                                4.616718e+08
     ERC20 min val rec                                      2.850451e+08
    max value received                                      1.692294e+08
    max val sent                                            4.394646e+07
     ERC20 total Ether sent contract                        3.756017e+07
    avg val received                                        8.323238e+06
    total transactions (including tnx to create contract    1.828997e+06
    Received Tnx                                            8.851734e+05
    Sent tnx                                                5.733918e+05
     Total ERC20 tnxs                                       2.002821e+05
    min value received                                      1.062298e+05
    Unique Received From Addresses                          8.917457e+04
    Unique Sent To Addresses                                6.960121e+04
    avg val sent                                            5.715935e+04
    Number of Created Contracts                             2.000685e+04
    min val sent                                            1.921264e+04
     ERC20 uniq sent addr                                   1.107809e+04
     ERC20 uniq rec addr                                    6.694262e+03
     ERC20 uniq rec contract addr                           2.974444e+02
     ERC20 uniq rec token name                              2.781759e+02
     ERC20 uniq sent token name                             4.536185e+01
    FLAG                                                    1.724110e-01
     ERC20 uniq sent addr.1                                 4.316210e-03
    max val sent to contract                                2.660652e-07
    total ether sent contracts                              2.660625e-07
    avg value sent to contract                              1.046096e-07
    min value sent to contract                              5.080371e-08
     ERC20 avg time between rec tnx                         0.000000e+00
     ERC20 avg time between rec 2 tnx                       0.000000e+00
     ERC20 avg time between contract tnx                    0.000000e+00
     ERC20 avg time between sent tnx                        0.000000e+00
     ERC20 min val sent contract                            0.000000e+00
     ERC20 max val sent contract                            0.000000e+00
     ERC20 avg val sent contract                            0.000000e+00
    dtype: float64



Investigando la varianza de las variables, se observó que hay algunas características con una varianza igual a 0.

- ERC20 avg time between rec tnx     
- ERC20 avg time between rec 2 tnx   
- ERC20 avg time between contract tnx
- ERC20 avg time between sent tnx    
- ERC20 min val sent contract        
- ERC20 max val sent contract        
- ERC20 avg val sent contract        

#### Distribucion variable objetivo


```python
print(df['FLAG'].value_counts())

pie, ax = plt.subplots(figsize=[8,6])
labels = ['No fraudulentas', 'Fraude']
#colors = ['#f9ae35', '#f64e38']
plt.pie(x = df['FLAG'].value_counts(),labels=labels, autopct='%.2f%%')
plt.title('Distribucion de la Variable obetjivo')
plt.show()
```

    FLAG
    0    7662
    1    2179
    Name: count, dtype: int64



    
![png](final_files/final_21_1.png)
    


#### Matriz de correlacion


```python
# Convertir columnas categóricas a variables dummy
df_with_dummies = pd.get_dummies(df, drop_first=True)

# Seleccionar solo las columnas numéricas nuevamente
numeric_df_with_dummies = df_with_dummies.select_dtypes(include=['float64', 'int64'])

# Calcular y visualizar la matriz de correlación
correlation_matrix = numeric_df_with_dummies.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5, fmt=".2f", square=True, center=0)
plt.title('Matriz de Correlación', fontsize=16)
plt.show()
```


    
![png](final_files/final_23_0.png)
    


### Limpieza de datos

#### Datos faltantes


```python
# Visualize missings pattern of the dataframe
plt.figure(figsize=(15,6))
sns.heatmap(df.isnull(), cbar=False)
plt.show()
```


    
![png](final_files/final_26_0.png)
    



```python
# Dropeamos las 2 variables categoricas
df.drop(df[categories], axis=1, inplace=True)
```


```python
# Reemplazamos datos numericos faltantes con la mediana
df.fillna(df.median(), inplace=True)
```


```python
# Volvemos a visualizar los valores faltantes
print(df.shape)
plt.figure(figsize=(15,6))
sns.heatmap(df.isnull(), cbar=False)
plt.show()
```

    (9841, 46)



    
![png](final_files/final_29_1.png)
    


#### Varianza = 0


```python
# variables con varianza = 0 
no_var = df.var() == 0
print(df.var()[no_var])
print('\n')
```

    ERC20 avg time between sent tnx        0.0
    ERC20 avg time between rec tnx         0.0
    ERC20 avg time between rec 2 tnx       0.0
    ERC20 avg time between contract tnx    0.0
    ERC20 min val sent contract            0.0
    ERC20 max val sent contract            0.0
    ERC20 avg val sent contract            0.0
    dtype: float64
    
    


Estas características no ayudarán en el rendimiento del modelo


```python
# Dropeamos las variables con var = 0
df.drop(df.var()[no_var].index, axis = 1, inplace = True)
print(df.var())
print(df.shape)
```

    FLAG                                                    1.724110e-01
    Avg min between sent tnx                                4.616718e+08
    Avg min between received tnx                            5.327656e+08
    Time Diff between first and last (Mins)                 1.042889e+11
    Sent tnx                                                5.733918e+05
    Received Tnx                                            8.851734e+05
    Number of Created Contracts                             2.000685e+04
    Unique Received From Addresses                          8.917457e+04
    Unique Sent To Addresses                                6.960121e+04
    min value received                                      1.062298e+05
    max value received                                      1.692294e+08
    avg val received                                        8.323238e+06
    min val sent                                            1.921264e+04
    max val sent                                            4.394646e+07
    avg val sent                                            5.715935e+04
    min value sent to contract                              5.080371e-08
    max val sent to contract                                2.660652e-07
    avg value sent to contract                              1.046096e-07
    total transactions (including tnx to create contract    1.828997e+06
    total Ether sent                                        1.283952e+11
    total ether received                                    1.326451e+11
    total ether sent contracts                              2.660625e-07
    total ether balance                                     5.877009e+10
     Total ERC20 tnxs                                       1.835047e+05
     ERC20 total Ether received                             1.017063e+20
     ERC20 total ether sent                                 1.275951e+18
     ERC20 total Ether sent contract                        3.439675e+07
     ERC20 uniq sent addr                                   1.014723e+04
     ERC20 uniq rec addr                                    6.133643e+03
     ERC20 uniq sent addr.1                                 3.953491e-03
     ERC20 uniq rec contract addr                           2.735599e+02
     ERC20 min val rec                                      2.610488e+08
     ERC20 max val rec                                      1.016835e+20
     ERC20 avg val rec                                      4.198599e+16
     ERC20 min val sent                                     1.016499e+12
     ERC20 max val sent                                     1.274901e+18
     ERC20 avg val sent                                     3.203738e+17
     ERC20 uniq sent token name                             4.168819e+01
     ERC20 uniq rec token name                              2.558699e+02
    dtype: float64
    (9841, 39)


#### Correlacion entre variables


```python
# volvemos a ver la matriz de correlacion
corr = df.corr()

with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(12,10))
    #sns.heatmap(corr,  mask=mask, annot=False, cmap='CMRmap', center=0, linewidths=0.1, square=True)
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, fmt=".2f", square=True, center=0)

```


    
![png](final_files/final_35_0.png)
    


dropeamos las variables que estan muy correlacionadas entre si:


```python
drop = ['total transactions (including tnx to create contract', 'total ether sent contracts', 'max val sent to contract', ' ERC20 avg val rec',
        ' ERC20 avg val rec',' ERC20 max val rec', ' ERC20 min val rec', ' ERC20 uniq rec contract addr', 'max val sent', ' ERC20 avg val sent',
        ' ERC20 min val sent', ' ERC20 max val sent', ' Total ERC20 tnxs', 'avg value sent to contract', 'Unique Sent To Addresses',
        'Unique Received From Addresses', 'total ether received', ' ERC20 uniq sent token name', 'min value received', 'min val sent', ' ERC20 uniq rec addr' ]
df.drop(drop, axis=1, inplace=True)
df.columns
```




    Index(['FLAG', 'Avg min between sent tnx', 'Avg min between received tnx',
           'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
           'Number of Created Contracts', 'max value received ',
           'avg val received', 'avg val sent', 'min value sent to contract',
           'total Ether sent', 'total ether balance',
           ' ERC20 total Ether received', ' ERC20 total ether sent',
           ' ERC20 total Ether sent contract', ' ERC20 uniq sent addr',
           ' ERC20 uniq sent addr.1', ' ERC20 uniq rec token name'],
          dtype='object')




```python
# volvemos a ver la matriz de correlacion
corr = df.corr()

with sns.axes_style('white'):
    fig, ax = plt.subplots(figsize=(12,10))
    #sns.heatmap(corr,  mask=mask, annot=False, cmap='CMRmap', center=0, linewidths=0.1, square=True)
    sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5, fmt=".2f", square=True, center=0)

```


    
![png](final_files/final_38_0.png)
    


#### Distribucion de variables restantes


```python
# variables finales:
columns = df.columns
columns
```




    Index(['FLAG', 'Avg min between sent tnx', 'Avg min between received tnx',
           'Time Diff between first and last (Mins)', 'Sent tnx', 'Received Tnx',
           'Number of Created Contracts', 'max value received ',
           'avg val received', 'avg val sent', 'min value sent to contract',
           'total Ether sent', 'total ether balance',
           ' ERC20 total Ether received', ' ERC20 total ether sent',
           ' ERC20 total Ether sent contract', ' ERC20 uniq sent addr',
           ' ERC20 uniq sent addr.1', ' ERC20 uniq rec token name'],
          dtype='object')




```python
fig, axes = plt.subplots(6, 3, figsize=(16, 20), constrained_layout =True)
plt.subplots_adjust(wspace = 0.7, hspace=2)

ax = sns.boxplot(ax = axes[0,0], data=df, x=columns[1])
ax1 = sns.boxplot(ax = axes[0,1], data=df, x=columns[2])
ax2 = sns.boxplot(ax = axes[0,2], data=df, x=columns[3])
ax3 = sns.boxplot(ax = axes[1,0], data=df, x=columns[4])
ax4 = sns.boxplot(ax = axes[1,1], data=df, x=columns[5])
ax5 = sns.boxplot(ax = axes[1,2], data=df, x=columns[6])
ax6 = sns.boxplot(ax = axes[2,0], data=df, x=columns[7])
ax7 = sns.boxplot(ax = axes[2,1], data=df, x=columns[8])
ax8 = sns.boxplot(ax = axes[2,2], data=df, x=columns[9])
ax9 = sns.boxplot(ax = axes[3,0], data=df, x=columns[10])
ax10 = sns.boxplot(ax = axes[3,1], data=df, x=columns[11])
ax11 = sns.boxplot(ax = axes[3,2], data=df, x=columns[12])
ax12 = sns.boxplot(ax = axes[4,0], data=df, x=columns[13])
ax13 = sns.boxplot(ax = axes[4,1], data=df, x=columns[14])
ax14 = sns.boxplot(ax = axes[4,2], data=df, x=columns[15])
ax15 = sns.boxplot(ax = axes[5,0], data=df, x=columns[16])
ax16 = sns.boxplot(ax = axes[5,1], data=df, x=columns[17])
ax17 = sns.boxplot(ax = axes[5,2], data=df, x=columns[18])
plt.show()
```


    
![png](final_files/final_41_0.png)
    



```python
# Variables con distribuciones pequenas
for i in df.columns[1:]:
    if len(df[i].value_counts()) < 10:
        print(f'Distribucion de {i}:  \n{df[i].value_counts()}')
        print('--------------------------------------------------------')
```

    Distribucion de min value sent to contract:  
    min value sent to contract
    0.00    9839
    0.02       1
    0.01       1
    Name: count, dtype: int64
    --------------------------------------------------------
    Distribucion de  ERC20 uniq sent addr.1:  
     ERC20 uniq sent addr.1
    0.0    9813
    1.0      26
    3.0       1
    2.0       1
    Name: count, dtype: int64
    --------------------------------------------------------


Se puede observar que los valores de estas dos variables en su mayoría son 0. por lo tanto, ambas variables serán descartadas ya que no serán útiles para nuestro modelo


```python
drops = ['min value sent to contract', ' ERC20 uniq sent addr.1']
df.drop(drops, axis=1, inplace=True)
print(df.shape)
df.head()
```

    (9841, 17)





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
      <th>FLAG</th>
      <th>Avg min between sent tnx</th>
      <th>Avg min between received tnx</th>
      <th>Time Diff between first and last (Mins)</th>
      <th>Sent tnx</th>
      <th>Received Tnx</th>
      <th>Number of Created Contracts</th>
      <th>max value received</th>
      <th>avg val received</th>
      <th>avg val sent</th>
      <th>total Ether sent</th>
      <th>total ether balance</th>
      <th>ERC20 total Ether received</th>
      <th>ERC20 total ether sent</th>
      <th>ERC20 total Ether sent contract</th>
      <th>ERC20 uniq sent addr</th>
      <th>ERC20 uniq rec token name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>844.26</td>
      <td>1093.71</td>
      <td>704785.63</td>
      <td>721</td>
      <td>89</td>
      <td>0</td>
      <td>45.806785</td>
      <td>6.589513</td>
      <td>1.200681</td>
      <td>865.691093</td>
      <td>-279.224419</td>
      <td>3.558854e+07</td>
      <td>3.560317e+07</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>12709.07</td>
      <td>2958.44</td>
      <td>1218216.73</td>
      <td>94</td>
      <td>8</td>
      <td>0</td>
      <td>2.613269</td>
      <td>0.385685</td>
      <td>0.032844</td>
      <td>3.087297</td>
      <td>-0.001819</td>
      <td>4.034283e+02</td>
      <td>2.260809e+00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>246194.54</td>
      <td>2434.02</td>
      <td>516729.30</td>
      <td>2</td>
      <td>10</td>
      <td>0</td>
      <td>1.165453</td>
      <td>0.358906</td>
      <td>1.794308</td>
      <td>3.588616</td>
      <td>0.000441</td>
      <td>5.215121e+02</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>10219.60</td>
      <td>15785.09</td>
      <td>397555.90</td>
      <td>25</td>
      <td>9</td>
      <td>0</td>
      <td>500.000000</td>
      <td>99.488840</td>
      <td>70.001834</td>
      <td>1750.045862</td>
      <td>-854.646303</td>
      <td>1.711105e+04</td>
      <td>1.141223e+04</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>36.61</td>
      <td>10707.77</td>
      <td>382472.42</td>
      <td>4598</td>
      <td>20</td>
      <td>1</td>
      <td>12.802411</td>
      <td>2.671095</td>
      <td>0.022688</td>
      <td>104.318883</td>
      <td>-50.896986</td>
      <td>1.628297e+05</td>
      <td>1.235399e+05</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>27.0</td>
    </tr>
  </tbody>
</table>
</div>



## Preparacion de los datos


```python
y = df.iloc[:, 0] # variable objetivo a predecir
X = df.iloc[:, 1:] # variables predictoras
print("X:", X.shape, "\ny:", y.shape)
```

    X: (9841, 16) 
    y: (9841,)



```python
# train y test split (80, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 666)
print("X_train:", X_train.shape, "\ny_train:",y_train.shape)
print("X_test:", X_test.shape, "\ny_test:", y_test.shape)
```

    X_train: (7872, 16) 
    y_train: (7872,)
    X_test: (1969, 16) 
    y_test: (1969,)


#### Normalizacion


```python
norm = PowerTransformer()
norm_train_f = norm.fit_transform(X_train)
```


```python
norm_df = pd.DataFrame(norm_train_f, columns=X_train.columns)
norm_df
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
      <th>Avg min between sent tnx</th>
      <th>Avg min between received tnx</th>
      <th>Time Diff between first and last (Mins)</th>
      <th>Sent tnx</th>
      <th>Received Tnx</th>
      <th>Number of Created Contracts</th>
      <th>max value received</th>
      <th>avg val received</th>
      <th>avg val sent</th>
      <th>total Ether sent</th>
      <th>total ether balance</th>
      <th>ERC20 total Ether received</th>
      <th>ERC20 total ether sent</th>
      <th>ERC20 total Ether sent contract</th>
      <th>ERC20 uniq sent addr</th>
      <th>ERC20 uniq rec token name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.098362</td>
      <td>-1.180122</td>
      <td>-1.636972</td>
      <td>-1.391535</td>
      <td>-1.784446</td>
      <td>-0.402604</td>
      <td>-1.399091</td>
      <td>-1.275325</td>
      <td>-1.131648</td>
      <td>-1.244257</td>
      <td>-0.007148</td>
      <td>-0.742358</td>
      <td>-0.406252</td>
      <td>-0.037421</td>
      <td>-0.432068</td>
      <td>0.235920</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-1.098362</td>
      <td>0.995731</td>
      <td>0.588180</td>
      <td>-1.391535</td>
      <td>0.600046</td>
      <td>2.483708</td>
      <td>0.766258</td>
      <td>0.847762</td>
      <td>-1.131648</td>
      <td>-1.244257</td>
      <td>-0.006259</td>
      <td>-0.742358</td>
      <td>-0.406252</td>
      <td>-0.037421</td>
      <td>-0.432068</td>
      <td>-0.991948</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.055605</td>
      <td>-1.180122</td>
      <td>-1.160854</td>
      <td>0.038133</td>
      <td>-0.999103</td>
      <td>-0.402604</td>
      <td>1.097323</td>
      <td>1.446629</td>
      <td>1.209193</td>
      <td>0.791664</td>
      <td>-0.007148</td>
      <td>-0.742358</td>
      <td>-0.406252</td>
      <td>-0.037421</td>
      <td>-0.432068</td>
      <td>-0.991948</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.342128</td>
      <td>0.433528</td>
      <td>1.053864</td>
      <td>1.506251</td>
      <td>0.630062</td>
      <td>-0.402604</td>
      <td>0.049769</td>
      <td>-0.540027</td>
      <td>-0.477909</td>
      <td>0.685250</td>
      <td>-0.007359</td>
      <td>1.754996</td>
      <td>2.530393</td>
      <td>-0.037421</td>
      <td>2.417875</td>
      <td>1.538489</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.098362</td>
      <td>0.643816</td>
      <td>0.768595</td>
      <td>-1.391535</td>
      <td>1.345826</td>
      <td>2.483708</td>
      <td>-1.122742</td>
      <td>-1.071357</td>
      <td>-1.131648</td>
      <td>-1.244257</td>
      <td>-0.007073</td>
      <td>-0.202641</td>
      <td>-0.406252</td>
      <td>-0.037421</td>
      <td>-0.432068</td>
      <td>0.708753</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>7867</th>
      <td>-1.098362</td>
      <td>1.294629</td>
      <td>0.287463</td>
      <td>-1.391535</td>
      <td>-0.611048</td>
      <td>2.483708</td>
      <td>0.520051</td>
      <td>0.709644</td>
      <td>-1.131648</td>
      <td>-1.244257</td>
      <td>-0.007066</td>
      <td>-0.742358</td>
      <td>-0.406252</td>
      <td>-0.037421</td>
      <td>-0.432068</td>
      <td>-0.991948</td>
    </tr>
    <tr>
      <th>7868</th>
      <td>-1.098362</td>
      <td>-1.180122</td>
      <td>-1.636972</td>
      <td>-1.391535</td>
      <td>-1.784446</td>
      <td>-0.402604</td>
      <td>-1.399091</td>
      <td>-1.275325</td>
      <td>-1.131648</td>
      <td>-1.244257</td>
      <td>-0.007148</td>
      <td>-0.742358</td>
      <td>-0.406252</td>
      <td>-0.037421</td>
      <td>-0.432068</td>
      <td>0.235920</td>
    </tr>
    <tr>
      <th>7869</th>
      <td>1.408298</td>
      <td>0.502400</td>
      <td>0.600756</td>
      <td>0.874752</td>
      <td>0.494093</td>
      <td>-0.402604</td>
      <td>1.283320</td>
      <td>1.266308</td>
      <td>1.306751</td>
      <td>1.313188</td>
      <td>-0.007148</td>
      <td>1.838631</td>
      <td>2.530835</td>
      <td>-0.037421</td>
      <td>2.346354</td>
      <td>1.643084</td>
    </tr>
    <tr>
      <th>7870</th>
      <td>-0.544641</td>
      <td>-1.180122</td>
      <td>-1.405601</td>
      <td>0.038133</td>
      <td>-0.611048</td>
      <td>-0.402604</td>
      <td>1.716349</td>
      <td>1.798433</td>
      <td>1.826996</td>
      <td>1.574253</td>
      <td>-0.007148</td>
      <td>-0.742358</td>
      <td>-0.406252</td>
      <td>-0.037421</td>
      <td>-0.432068</td>
      <td>-0.991948</td>
    </tr>
    <tr>
      <th>7871</th>
      <td>0.630945</td>
      <td>-1.110118</td>
      <td>-0.892940</td>
      <td>-0.202942</td>
      <td>-0.611048</td>
      <td>-0.402604</td>
      <td>1.026378</td>
      <td>1.270867</td>
      <td>1.332874</td>
      <td>0.791666</td>
      <td>-0.007148</td>
      <td>-0.742358</td>
      <td>-0.406252</td>
      <td>-0.037421</td>
      <td>-0.432068</td>
      <td>-0.991948</td>
    </tr>
  </tbody>
</table>
<p>7872 rows × 16 columns</p>
</div>



### Class Imbalance

SE USARA OVERSAMPLING DE SMOTE POR SU FACILIDAD


```python
y.value_counts(normalize=True) * 100 

# PORCENTAJE DE LA VARIABLE OBJETIVO
```




    FLAG
    0    77.857941
    1    22.142059
    Name: proportion, dtype: float64




```python
oversample = SMOTE()
x_tr_resample, y_tr_resample = oversample.fit_resample(norm_train_f, y_train)

print(f'antes DE SMOTE: {norm_train_f.shape, y_train.shape}')
print(f'despues de SMOTE: {x_tr_resample.shape, y_tr_resample.shape}')
```

    antes DE SMOTE: ((7872, 16), (7872,))
    despues de SMOTE: ((12208, 16), (12208,))


## Modelos

### Regresion Logistica


```python
LR = LogisticRegression(random_state=333)
LR.fit(x_tr_resample, y_tr_resample)

# Transform test features
norm_test_f = norm.transform(X_test)

preds = LR.predict(norm_test_f)
```


```python
print(y_test.shape)
y_test.value_counts()
```

    (1969,)





    FLAG
    0    1558
    1     411
    Name: count, dtype: int64



#### Metricas


```python
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))
```

                  precision    recall  f1-score   support
    
               0       0.96      0.89      0.92      1558
               1       0.67      0.86      0.75       411
    
        accuracy                           0.88      1969
       macro avg       0.82      0.88      0.84      1969
    weighted avg       0.90      0.88      0.89      1969
    
    [[1385  173]
     [  57  354]]


#### Matriz de confusion


```python
# Calcular la matriz de confusión
cm = confusion_matrix(y_test, preds)

# Crear el objeto ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LR.classes_)

# Configurar el tamaño de la figura
plt.figure(figsize=(8, 6))

# Plotear la matriz de confusión
disp.plot(cmap='Blues', values_format='d')

# Añadir título
plt.title('Matriz de Confusión')

# Mostrar la gráfica
plt.show()
```


    <Figure size 800x600 with 0 Axes>



    
![png](final_files/final_62_1.png)
    


Considerando la matriz de confusión:

- El modelo LR identificó correctamente 354 (TP) casos de FRAUDE, de un total de 411 (P).
- El modelo LR marcó como FRAUDE 173 (FP) casos de 1558, cuando estos casos eran en realidad NO-FRAUDE.
- En un escenario de detección de fraudes, nos importa más las transacciones que eran realmente FRAUDES, pero que fueron clasificadas como NO-FRAUDE por nuestro modelo (FN - 57) -------> ERROR TIPO II.

Por lo tanto, vamos a intentar aumentar el recall

### Random Forest


```python
RF = RandomForestClassifier(random_state=555)
RF.fit(x_tr_resample, y_tr_resample)
preds_RF = RF.predict(norm_test_f)
```

#### Metricas


```python
print(classification_report(y_test, preds_RF))
print(confusion_matrix(y_test, preds_RF))
```

                  precision    recall  f1-score   support
    
               0       0.98      0.98      0.98      1558
               1       0.93      0.93      0.93       411
    
        accuracy                           0.97      1969
       macro avg       0.96      0.95      0.96      1969
    weighted avg       0.97      0.97      0.97      1969
    
    [[1531   27]
     [  30  381]]


#### Matriz de confusion


```python
# Calcular la matriz de confusión
cm = confusion_matrix(y_test, preds_RF)

# Crear el objeto ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LR.classes_)

# Configurar el tamaño de la figura
plt.figure(figsize=(8, 6))

# Plotear la matriz de confusión
disp.plot(cmap='Blues', values_format='d')

# Añadir título
plt.title('Matriz de Confusión')

# Mostrar la gráfica
plt.show()
```


    <Figure size 800x600 with 0 Axes>



    
![png](final_files/final_69_1.png)
    


El clasificador Random forest, parece producir resultados más efectivos.

- Tanto los FP como los FN se reducen considerablemente, aumentando el recall y la precisión.
- Usando Random forest, el modelo no detecta 30 casos de FRAUDE.

### XGBoost Classifier



```python
xgb_c = xgb.XGBClassifier(random_state=888)
xgb_c.fit(x_tr_resample, y_tr_resample)
preds_xgb = xgb_c.predict(norm_test_f)
```

#### Metricas


```python
print(classification_report(y_test, preds_xgb))
print(confusion_matrix(y_test, preds_xgb))
```

                  precision    recall  f1-score   support
    
               0       0.99      0.99      0.99      1558
               1       0.94      0.95      0.95       411
    
        accuracy                           0.98      1969
       macro avg       0.97      0.97      0.97      1969
    weighted avg       0.98      0.98      0.98      1969
    
    [[1535   23]
     [  21  390]]


#### Matriz de confusion


```python
cm = confusion_matrix(y_test, preds_xgb)

# Crear el objeto ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LR.classes_)

# Configurar el tamaño de la figura
plt.figure(figsize=(8, 6))

# Plotear la matriz de confusión
disp.plot(cmap='Blues', values_format='d')

# Añadir título
plt.title('Matriz de Confusión')

# Mostrar la gráfica
plt.show()
```


    <Figure size 800x600 with 0 Axes>



    
![png](final_files/final_76_1.png)
    


Los resultados del XGBoost muestran que está obteniendo resultados ligeramente mejores que el Random Forest en lo que respecta a las transacciones NO-FRAUDE, marcando 23 casos como fraude cuando en realidad eran no-fraude.

En cuanto a la identificación de FRAUDES, el XGBoost dejó de detectar 21 transacciones de un total de 411, lo que sugiere el mejor puntaje de recall.

Considerando todo lo anterior, el mejor modelo es el XGBoost y es la opción que queremos tomar y mejorar si es posible

#### Parameter tunning

##### Grid search


```python
params_grid = {'learning_rate':[0.01, 0.1, 0.5],
              'n_estimators':[100,200],
              'subsample':[0.3, 0.5, 0.9],
               'max_depth':[2,3,4],
               'colsample_bytree':[0.3,0.5,0.7]}

grid = GridSearchCV(estimator=xgb_c, param_grid=params_grid, scoring='recall', cv = 10, verbose = 0)

grid.fit(x_tr_resample, y_tr_resample)
print(f'Best params found for XGBoost are: {grid.best_params_}')
print(f'Best recall obtained by the best params: {grid.best_score_}')
```

    Best params found for XGBoost are: {'colsample_bytree': 0.5, 'learning_rate': 0.5, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.5}
    Best recall obtained by the best params: 0.9900093906790802


##### Metricas


```python
preds_best_xgb = grid.best_estimator_.predict(norm_test_f)
print(classification_report(y_test, preds_best_xgb))
print(confusion_matrix(y_test, preds_best_xgb))
```

                  precision    recall  f1-score   support
    
               0       0.98      0.99      0.99      1558
               1       0.95      0.94      0.94       411
    
        accuracy                           0.98      1969
       macro avg       0.97      0.96      0.96      1969
    weighted avg       0.98      0.98      0.98      1969
    
    [[1536   22]
     [  24  387]]


##### Matriz de confusion


```python
# Calcular la matriz de confusión
cm = confusion_matrix(y_test, preds_best_xgb)

# Crear el objeto ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LR.classes_)

# Configurar el tamaño de la figura
plt.figure(figsize=(8, 6))

# Plotear la matriz de confusión
disp.plot(cmap='Blues', values_format='d')

# Añadir título
plt.title('Matriz de Confusión')

# Mostrar la gráfica
plt.show()
```


    <Figure size 800x600 with 0 Axes>



    
![png](final_files/final_85_1.png)
    


La matriz de confusión no muestra mejora; los resultados son muy similares a los obtenidos con el modelo sin ajustar los parametros con grid search

##### Curva AUC


```python
# Plotting AUC for untuned XGB Classifier
probs = xgb_c.predict_proba(norm_test_f)
pred = probs[:,1]
fpr, tpr, threshold = roc_curve(y_test, pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12,8))
plt.title('ROC for tuned XGB Classifier')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0,1], [0,1], 'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
```


    
![png](final_files/final_88_0.png)
    


##### Guardamos el modelo


```python
# Save the model for further use
pickle_out = open('XGB_FRAUD_ETH.pickle', 'wb')
pickle.dump(xgb_c, pickle_out)
pickle_out.close()
```
