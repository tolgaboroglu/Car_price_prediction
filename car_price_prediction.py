import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
import plotly.express as px

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

print(check_df(cars))

# ENCODING

# One Hot Encoding

ohe= OneHotEncoder()

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cars = one_hot_encoder(cars, cat_cols, drop_first=True)
print(cars.head())


# MODEL BUILDING

print(cars.columns)
X = cars.drop(['ID', 'Levy', 'Manufacturer', 'Model', 'Prod. year',
              'Category', 'Engine volume', 'Mileage', 'Cylinders', 'Color', 'Airbags',
              'Leather interior_Yes', 'Fuel type_Diesel', 'Fuel type_Hybrid',
              'Fuel type_Hydrogen', 'Fuel type_LPG', 'Fuel type_Petrol',
              'Fuel type_Plug-in Hybrid', 'Gear box type_Manual',
              'Gear box type_Tiptronic', 'Gear box type_Variator',
              'Drive wheels_Front', 'Drive wheels_Rear', 'Doors_04-May', 'Doors_>5',
              'Wheel_Right-hand drive'], axis  =1)

y = cars['Price']

X_train , X_test , y_train , y_test = train_test_split(X,y, test_size=0.3, shuffle=True, random_state=11)


# Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)


print(X_train)

print("X_train shape: {}".format(X_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_train shape: {}".format(y_train.shape))
print("y_test shape: {}".format(y_test.shape))


# Fitting Model

# LINEAR REGRESSION

model = LinearRegression()
print(model.fit(X_train, y_train))

y_pred = model.predict(X_test)
pd.DataFrame({'test':y_test, 'pred': y_pred}).head()

# Evaluation

print(f"MAE:{mean_absolute_error(y_test, y_pred)}")
print(f"RMSE:{mean_absolute_error(y_test, y_pred)}")

print(model.score(X_test,y_test))

# MAE:136314.46503538327
# RMSE:136314.46503538327
# -130.9646665492095

sns.regplot(x=y_test, y=y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.show()


# Decision Tree Regressor

dtr = DecisionTreeRegressor()
print(dtr.fit(X_train, y_train))

print(dtr.score(X_train, y_train))

print(dtr.score(X_test, y_test))

# train : 1.0
# test : -69.53659132560271


# RANDOM FOREST REGRESSOR

rfr = RandomForestRegressor()
print(rfr.fit(X_train, y_train))

print(rfr.score(X_train, y_train))

print(rfr.score(X_test, y_test))
# train : 0.9314720050029666
# test : -52.653368490680876


# GRADIENT BOOST REGRESSOR

gbr = GradientBoostingRegressor()
print(gbr.fit(X_train,y_train))

print(gbr.score(X_train, y_train))

# train : 0.9999973838353556

print((gbr.score(X_test,y_test)))

# test :  -69.34814546475437



models = pd.DataFrame({

    'Model' : ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boost'],
    'Score' : [model.score(X_test,y_test),dtr.score(X_test,y_test),rfr.score(X_test,y_test),gbr.score(X_test,y_test)]

})

#                Model       Score
# 2      Random Forest  -53.600922
# 3     Gradient Boost  -69.348145
# 1      Decision Tree  -69.536591
# 0  Linear Regression -130.964667

print(models.sort_values(by = 'Score', ascending=False))

px.bar(data_frame = models , x = 'Score', y = 'Model', color='Score', template='plotly_dark',
       title = 'Models Comparision')

