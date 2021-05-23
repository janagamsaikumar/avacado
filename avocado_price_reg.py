import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
dataset=pd.read_csv(r'C:\Users\saikumar\Desktop\AMXWAM data science\class 20 _oct 31,2020\TASK-22_ 4 Imp Sub Tasks\avocado.csv')
dataset.info()
dataset.shape
dataset.isnull().any() # no null values in the dataset 
dataset.head()
dataset=dataset.drop(columns=['Unnamed: 0','4046','4225','4770','Date'])
 # previously we have 18249 rows and 14 columns
 # we have dropped the irrelavent atttributes.
 len(dataset)
  # first use case is region with lowest and highest price of avocado
def get_average(dataset,columns):
    return sum(dataset[columns])/len(dataset)
def get_average_between_two_columns(dataset,region,AveragePrice):
    List=list(dataset[region].unique())
    average=[]

    for i in List:
        x=dataset[dataset[region]==i]
        region_average= get_average(x,AveragePrice)
        average.append(region_average)

    dataset_region_AveragePrice=pd.DataFrame({'region':List,'AveragePrice':average})
    region_AveragePrice_sorted_index=dataset_region_AveragePrice.AveragePrice.sort_values(ascending=True).index.values
    region_AveragePrice_sorted_data=dataset_region_AveragePrice.reindex(region_AveragePrice_sorted_index)
    return region_AveragePrice_sorted_data
def plot(data,xlabel,ylabel):
    plt.figure(figsize=(15,5))
    ax=sns.barplot(x=data.region,y=data.AveragePrice,palette='rocket')
    plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(('Avarage '+ylabel+' of Avocado According to '+xlabel));

data1=get_average_between_two_columns(dataset,'region','AveragePrice')
plot(data1,'Region','Price($)')

# lowest and highest avocado region 
print(data1['region'].iloc[0])
print(data1['region'].iloc[-1])

#highest region of avocado production
data2 = get_average_between_two_columns(dataset,'region','Total Volume')
sns.boxplot(x=data2.AveragePrice).set_title("Figure: Boxplot repersenting outlier columns.")

outlier_region = data2[data2.AveragePrice>10000000]
print(outlier_region['region'].iloc[-1], "is outlier value")

outlier_region.index
data2=data2.drop(outlier_region.index,axis=0)

plot(data2,'region','totalvolume')

#avgerage price of avocado in each year
data3 = get_average_between_two_columns(dataset,'year','AveragePrice')
plot(data3,'year','Price')
 

dataset.info()
#two categorical data
dataset['region'] = dataset['region'].astype('category') 
dataset['region'] = dataset['region'].cat.codes

dataset['type'] = dataset['type'].astype('category')
dataset['type'] = dataset['type'].cat.codes # converts only categorical into numerical 

dataset.info()

# split into X and y
X=dataset.iloc[:,1:]
y=dataset.iloc[:,-9]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)

from sklearn.metrics import r2_score
ac=r2_score(y_test,y_pred)
print(ac*100)


                                                    



