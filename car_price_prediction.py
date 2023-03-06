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

# NUMERICAL AND CATEGORICAL DATA

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "0"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                dataframe[col].dtypes != "0"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                dataframe[col].dtypes == "0"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observations:{dataframe.shape[0]}")
    print(f"Variables:{dataframe.shape[1]}")
    print(f'cat_cols:"{len(cat_cols)}')
    print(f'num_cols:"{len(num_cols)}')
    print(f'cat_but_car:"{len(cat_but_car)}')
    print(f'num_but_cat:"{len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(cars)



# numeric

def numSummary(dataframe, numericalCol, plot=False):
    quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1]
    print(dataframe[numericalCol].describe(quantiles).T)

    if plot:
        dataframe[numericalCol].hist()
        plt.xlabel(numericalCol)
        plt.title(numericalCol)
        plt.show(block=True)

for col in num_cols:
    print(f"{col}:")
    numSummary(cars, col, True)


# categorical

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    if cars[col].dtypes == "bool":
        print(col)
    else:
        cat_summary(cars, col, True)


# TARGET ANALYZE

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col:"mean"}), end="\n\n\n")

for col in num_cols:
        target_summary_with_num(cars,"Price",col)


# OUTLIERS

def outliers_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit , up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outliers_thresholds(dataframe, col_name)
    # aslında yukarıada yaptığımız any yani bool ile herhangi bir boş aykırı değer var mı sorusuna denk gelir
    if dataframe[(dataframe[col_name]>up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

    check_outlier(cars, "Price")

for col in num_cols:
    print(col, "=>", check_outlier(cars, col))

# price
# cylinders

def grab_outliers(dataframe, col_name, index = False):
    low,up = outliers_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name]>low) | (dataframe[col_name] < up)].shape[0] > 10:
        print(dataframe[(dataframe[col_name]<low)| (dataframe[col_name] > up)].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name]> up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name]< low) | (dataframe[col_name] > up))].index
        return outlier_index

    for col in num_cols:
        print(col, grab_outliers(cars, col, True))


# Eksik gözlem analizinin yapılması

def missing_values_table(dataframe,na_name = False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss,np.round(ratio,2)], axis=1, keys=['n_miss','ratio'])
    print(missing_df,end="\n")

    if na_name:
        return na_columns

missing_values_table(cars)

# CORRELATION



corr = cars[num_cols].corr()
print(corr)





sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(corr,cmap="RdBu")
plt.show(block=True)


# MISSING VALUES

# remove missing values

cars = cars.dropna()






