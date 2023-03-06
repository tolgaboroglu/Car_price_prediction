import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno


train = pd.read_csv("train.csv")
print((train.head()))

test = pd.read_csv("test.csv")
print(test.head())

##########################################

print(train.tail())

print(test.tail())

#########################################

print("######################## train summary ######################")

def check_df(train, head=5):
    print("####################### shape ##########################")

    print(train.shape)

    print("####################### types ##########################")

    print(train.dtypes)

    print("####################### head ##########################")

    print(train.head)

    print("####################### tail ##########################")

    print(train.tail)

    print("####################### NA ##########################")

    print(train.isnull().sum)

    print("##################### Quantiles #####################")

    print(train.quantile([0,0.05,0.50,0.95,0.99,1]).T)

print(check_df(train))


print("######################### test summary ######################")

def check_df(test, head=5):
    print("####################### shape ##########################")

    print(test.shape)

    print("####################### types ##########################")

    print(test.dtypes)

    print("####################### head ##########################")

    print(test.head)

    print("####################### tail ##########################")

    print(test.tail)

    print("####################### NA ##########################")

    print(test.isnull().sum)

    print("##################### Quantiles #####################")

    print(test.quantile([0,0.05,0.50,0.95,0.99,1]).T)

print(check_df(test))

##############################################################################

print("############################# merge dataset #########################")

cars = pd.concat([train, test], axis = 0 ,ignore_index= True)

print(cars.head())

print("######################## cars ##############################")
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

print(check_df(cars))

##################################################################################

# DATA CLEANING


print(msno.bar(cars,figsize=(15, 5),fontsize=10,color = '#459E97'))


