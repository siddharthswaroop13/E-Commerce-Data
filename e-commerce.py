# E-Commerce-Data
# Analysis of a transactions data set with a ML model that predicts sales.


###### E-COMMERCE SALES PREDICTION

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


## We import our data using .read_csv() method and we also add a parameter "encoding='latin'" as default encoding engine
# wasn't able to process this particular dataset. So next time you have difficulties importing data and everything seems
# to be correct and OK, check out encoding. That might save you some time of googling to try to understand what's wrong.

df = pd.read_csv(r'C:\\Users\\Siddharth\\desktop\\data.csv',engine='python',encoding='latin')
print(df)

## Just by looking at first 5 rows of our table we can understand the structure and datatypes present in our dataset.
# We can notice that we will have to deal with timeseries data, integers and floats, categorical and text data.
print(df.head())

## Exploratory data analysis
##Every data science project starts with EDA as we have to understand what do we have to deal with. I divide
# EDA into 2 types: visual and numerical. Let's start with numerical as the simple pndas method .describe()
# gives us a lot of useful information.

print(df.shape)

print(df.columns)
print("\n")

print(df.isnull().any())
print("\n")

print(df.isnull().values.any())
print("\n")

print(df.isnull().sum())
print("\n")

##Quick statistical overview

print(df.describe())

##Just a quick look at data with .describe() method gives us a lot of space to think. We see negative
# quantities and prices, we can see that not all records have CustomerID data, we can also see that the
# majority of transactions are for quantites from 3 to 10 items, majority of items have price up to 5 pounds
# and that we have a bunch of huge outliers we will have to deal with later.

## Dealing with types
#.read_csv() method performs basic type check, but it doesn't do that perfectly. That's why it is much better
# to deal with data types in our dataframe before any modifications to prevent additional difficulties. Every
# pandas dataframe has an attribute .dtypes which will help us understand what we currently have and what data
# has to be casted to correct types.

print(df.dtypes)
print("\n")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df = df.set_index('InvoiceDate')

## DEALING WITH NULL VALUES
## Next and very important step is dealing with missing values. Normally if you encounter null values
## in the dataset you have to understand nature of those null values and possible impact they could have
## on the model. There are few strategies that we can use to fix our issue with null values:
# delete rows with null values
# delete the feature with null values
# impute data with mean or median values or use another imputing strategy (method .fillna())
# Let's check out what we have here.

print(df.isnull().sum())
print("\n")

#CustomerID has too much null values and this feature cannot predict a lot so we can just drop it.
# Also it could be reasonable to create another feature "Amount of orders per customer", but.... next time ;)

df = df.drop(columns=['CustomerID'])

##Let's check out what kind of nulls we have in Description

print(df[df['Description'].isnull()].head())
print("\n")

##The data in these rows is pretty strange as UnitPrice is 0, so these orders do not generate any sales.
# I think, we can impute it with "UNKNOWN ITEM" at the moment and deal with those later during the analysis.

df['Description'] = df['Description'].fillna('UNKNOWN ITEM')
print(df.isnull().sum())


## Checking out columns separately
#Also it makes sense to go feature by feature and check what pitfalls we have in our data and also to understand our numbers better.

#Let's continue checking Description column. Here we can see items that were bought most often.

print(df['Description'].value_counts().head(10))
print("\n")

print(df['Description'].value_counts().tail(10))
print("\n")

print(df['Description'].unique())
print("\n")

print(df['Description'].sum())
print("\n")

print(len(df['Description'].unique()))

## Here we can see our best selling products, items that appear in orders the most often. Also to make it
# visually more appealing let's create a bar chart for 15 top items.

item_counts = df['Description'].value_counts().sort_values(ascending=False).iloc[0:15] ## or can use head(15) inplace of iloc[0:15] it just gives same result and graph
plt.figure(figsize=(18,6))
sns.barplot(item_counts.index, item_counts.values, palette=sns.cubehelix_palette(15))
plt.ylabel("Counts")
plt.title("Which items were bought more often?");
plt.xticks(rotation=90);
plt.show()

print(df['Description'].value_counts().tail())
##We also notice from above code that valid items are normally uppercased and non-valid or cancelations are in lower case

print(df[~df['Description'].str.isupper()]['Description'].value_counts().head())

##Quick check of the case of letters in Description says that there are some units with lower case letters
# in their name and also that lower case records are for canceled items. Here we can understand that data management
# in the store can be improved.

lcase_counts = df[~df['Description'].str.isupper()]['Description'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(18,6))
sns.barplot(lcase_counts.index, lcase_counts.values, palette=sns.color_palette("hls", 15))
plt.ylabel("Counts")
plt.title("Not full upper case items");
plt.xticks(rotation=90);
plt.show()

######### CORRELATION MATRIX AND PLOT.
corr_matrix = df.corr()

plt.subplots(figsize = (10,8))
sns.heatmap(corr_matrix, annot = True, cmap = "Blues")
plt.show()

## STOCKCODE FEATURE
#ALso checking out stoke codes, looks like they are deeply correlated with descriptions - which makes perfect sense.

print(df['StockCode'].value_counts().head(10))
print("\n")

print(df['StockCode'].value_counts().tail(10))
print("\n")

print(df['StockCode'].unique())
print("\n")

print(len(df['StockCode'].unique()))
print("\n")

stock_counts = df['StockCode'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(18,6))
sns.barplot(stock_counts.index, stock_counts.values, palette=sns.color_palette("GnBu_d"))
plt.ylabel("Counts")
plt.title("Which stock codes were used the most?");
plt.xticks(rotation=90);
plt.show()

stock_counts = df['StockCode'].value_counts().sort_values(ascending=False).tail(25)  ## or can use iloc also
plt.figure(figsize=(18,6))
sns.barplot(stock_counts.index, stock_counts.values, palette=sns.color_palette("GnBu_d"))
plt.ylabel("Counts")
plt.title("Which stock codes were used the least?");
plt.xticks(rotation=90);
plt.show()

########### LET'S EXPLORE INVOICE FEATURE
## Checking out also InvoiceNo feature.

print(df['InvoiceNo'].value_counts().tail())
print("\n")

print(df['InvoiceNo'].value_counts().head())
print("\n")

inv_counts = df['InvoiceNo'].value_counts().sort_values(ascending=False).iloc[0:15]
plt.figure(figsize=(18,6))
sns.barplot(inv_counts.index, inv_counts.values, palette=sns.color_palette("BuGn_d"))
plt.ylabel("Counts")
plt.title("Which invoices had the most items?");
plt.xticks(rotation=90);
plt.show()

inv_counts = df['InvoiceNo'].value_counts().sort_values(ascending=False).head(30)
plt.figure(figsize=(18,6))
sns.barplot(inv_counts.index, inv_counts.values, palette=sns.color_palette("BuGn_d"))
plt.ylabel("Counts")
plt.title("Which invoices had the most items?");
plt.xticks(rotation=90);
plt.show()

###### SOLVING THE MYSTERY OF CANCELED/RETURNED INVOICES HAVING NEGATIVE QUANTITIES.
##Looks like Invoices that start with 'C' are the "Canceling"/"Returning" invoices.
# This resolves the mistery with negative quantities.

print(df[df['InvoiceNo'].str.startswith('C')].describe())
print("\n")

print(df[df['InvoiceNo'].str.startswith('C')].info())
print("\n")

print(df[df['InvoiceNo'].str.startswith('C')])
print("\n")

print(len(df[df['InvoiceNo'].str.startswith('C')]))
print("\n")

#Although, we should've gotten deeper into analysis of those returns, for the sake of simplicity let's just ignore those values for the moment.
#We can actually start a separate project based on that data and predict the returning/cancelling rates for the store.

df = df[~df['InvoiceNo'].str.startswith('C')]
print(df.describe())
print("\n")

## During exploratory data analysis we can go back to the same operations and checks, just to understand
# how our actions affected the dataset. EDA is the series of repetitive tasks to understand better our data.
# And here, for example we get back to .describe() method to get an overall picture of our data after some manipulations.

#We still see negative quantities and negative prices, let's get into those records.

df[df['Quantity'] < 0]
print(df[df['Quantity'] < 0].head())

## Here we can see that other "Negative quantities" appear to be damaged/lost/unknown items.
# Again, we will just ignore them for the sake of simplicity of analysis for this project.

df = df[df['Quantity'] > 0]  ## or can also use alterntaley df[~df['Quantity'] < 0]
print(df.describe())

## We also see negative UnitPrice, which is not normal as well. Let's check this out.

print(df[df['UnitPrice'] < 0].describe())
print("\n")

## checking the specific gift having negative unit price of -11062.06

print(df[df['UnitPrice'] == -11062.06])
print("\n")

# As there are just two rows, let's ignore them for the moment (description gives us enough warnings,
# althoug we still need some context to understand it better)

df = df[df['UnitPrice'] > 0]   ## or can also use alterntaley df[~df['UnitPrice'] < 0]
print(df.describe())

#############################################################################################

## INTRODUCING/CREATING/MAKING NEW FEATURES IN OUR DATASET
#As we have finished cleaning our data and removed all suspicious records we can start creating some new
# features for our model. Let's start with the most obvious one - Sales. We have quantities,
# we have prices - we can calculate the revenue.

df['Sales'] = df['Quantity'] * df['UnitPrice']
print(df.head())

##########  Visual EDA ##############

plt.figure(figsize=(3,6))
sns.countplot(df[df['Country'] == 'United Kingdom']['Country'])
plt.xticks(rotation=90)
plt.show()

plt.figure(figsize=(18,6))
sns.countplot(df[df['Country'] != 'United Kingdom']['Country'])
plt.xticks(rotation=90)
plt.show()

#### CALCULATING THE SHARE/WEIGHTAGE/PORTION OF UK IN TOTAL GIFTS BOUGHT OR SENT OUT

uk_count = df[df['Country'] == 'United Kingdom']['Country'].count()
all_count = df['Country'].count()
uk_perc = uk_count/all_count
print("The percentage of UK share in total gifts sales : ", str('{0:.2f}%').format(uk_perc*100))
print("\n")
print(uk_count)
print("\n")
print(all_count)
print("\n")

##From above plots and calculations we can see that vast majority of sales were made in UK
# and just 8.49% went abroad. We can say our dataset is skewed to the UK side :D.

## DETECTING OUTLIERS IN DATASET

# There are few different methods to detect outliers: box plots, using IQR, scatter plot also
# works in some cases (and this is one of those). Also, detecting outliers using scatter plot
# is pretty intuitive. You plot your data and remove data points that visually are definitely
# out of range. Like in the chart below.

print(df.index.unique())
print(len(df.index.unique()))

plt.figure(figsize=(18,6))
plt.scatter(x=df.index, y=df['Sales'])
plt.show()

print(df.quantile([0.05, 0.95, 0.98, 0.99, 0.999]))

## We can see that if we remove top 2% of our data points we will get rid of absolute outliers and will have more balanced dataset.

df_quantile = df[df['Sales'] < 125]
plt.scatter(x=df_quantile.index, y=df_quantile['Sales'])
plt.xticks(rotation=90)
plt.show()

print(df_quantile.describe())

## Looks like our data is almost ready for modelling. We performed a clean up, we removed outliers that
# were disturbing the balance of our dataset, we removed invalid records -
# now our data looks much better! and it doesn't lose it's value.

## Visually checking distribution of numeric features

plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['UnitPrice'] < 10]['UnitPrice'].values, kde=True, bins=10)
plt.show()

plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['UnitPrice'] < 5]['UnitPrice'].values, kde=True, bins=10, color='green')
plt.show()

## From these histograms we can see that vast majority of items sold in this store has low price range - 0 to 3 pounds.

## Now we will be closely,keenly and deeply observing, looking and plotting for quantity column.
## and drawing some inference conclusion from it just like the above feature of UnitPrice.

plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['Quantity'] <= 30]['Quantity'], kde=True, bins=10, color='red')
plt.show()

plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['Quantity'] <= 15]['Quantity'], kde=True, bins=10, color='orange')
plt.show()

## Conclusion for quantity : From these histograms we that people bought normally 1-5 items or 10-12
# - maybe there were some kind of offers for sets?

### Looking briefly at SALES FEATURE NOW.

plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['Sales'] < 60]['Sales'], kde=True, bins=10, color='purple')
plt.show()

plt.figure(figsize=(12,4))
sns.distplot(df_quantile[df_quantile['Sales'] < 30]['Sales'], kde=True, bins=10, color='grey')
plt.show()

## From these histograms we can understand that majority of sales per order were in range 1-15 pounds each.

## Analysing sales over time

df_ts = df[['Sales']]
print(df_ts.head())

## As we can see every invoice has it's own timestamp (definitely based on time the order was made).
# We can resample time data by, for example weeks, and try see if there is any patterns in our sales.

plt.figure(figsize=(18,6))
df_resample = df_ts.resample('W').sum()
df_resample.plot()
plt.show()

plt.figure(figsize=(18,6))
df_resample = df_ts.resample('M').sum()
df_resample.plot()
plt.show()

## That week with 0 sales in January looks suspicious, let's check it closer

print(df_resample['12-2010':'01-2011'])

## Now it makes sense - possibly, during the New Year holidays period the store was closed and didn't process orders,
# that's why they didn't make any sales.


########### Preparing data for modeling and feature creation ####################################

## Now it comes the most fun part of the project - building a model. To do this we will need to create few more
# additional features to make our model more sophisticated.

df_clean = df[df['UnitPrice'] < 15]
print(df_clean.describe())
print("\n")

print(df_clean.index)

## Quantity per invoice feature

#A feature that could influence the sales output could be "Quantity per invoice". Let's find the data for this feature.

df_join = df_clean.groupby('InvoiceNo')[['Quantity']].sum()
print(df_join)
print("\n")

df_join = df_join.reset_index()
print(df_join.head())

df_clean['InvoiceDate'] = df_clean.index
df_clean = df_clean.merge(df_join, how='left', on='InvoiceNo')
print(df_clean.tail(15))

df_clean = df_clean.rename(columns={'Quantity_x' : 'Quantity', 'Quantity_y' : 'QuantityInv'})
print(df_clean.tail(15))

print(df_clean.describe())

df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
print(df_clean.dtypes)

## 4.2. Bucketizing Quantity and UnitPrice features¶
##Based on the EDA done previously we can group these features into 6 buckets
# for Quantity and 5 for UnitePrice using pandas .cut() method.

bins_q = pd.IntervalIndex.from_tuples([(0, 2), (2, 5), (5, 8), (8, 11), (11, 14), (15, 5000)])
df_clean['QuantityRange'] = pd.cut(df_clean['Quantity'], bins=bins_q)
bins_p = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3), (3, 4), (4, 20)])
df_clean['PriceRange'] = pd.cut(df_clean['UnitPrice'], bins=bins_p)
print(df_clean.head())

#### Extracting and bucketizing dates
## We have noticed that depends on a season gifts sell differently: pick of sales is in the Q4, then it
# drastically drops in Q1 of the next year and continues to grow till its new pick in Q4 again. From this
# observation we can create another feature that could improve our model.

df_clean['Month'] = df_clean['InvoiceDate'].dt.month
print(df_clean.head())

bins_d = pd.IntervalIndex.from_tuples([(0,3),(3,6),(6,9),(9,12)])
df_clean['DateRange'] = pd.cut(df_clean['Month'], bins=bins_d, labels=['q1','q2','q3','q4'])
print(df_clean.tail())

########################################################################################################################

### Building a model
### Splitting data into UK and non-UK
### We have to analyze these 2 datasets separately to have more standardized data for a model, because
# there can be some patterns that work for other countries and do not for UK or vise versa. Also a hypothesis
# to test - does the model built for UK performs good on data for other countries?

df_uk = df_clean[df_clean['Country'] == 'United Kingdom']
df_abroad = df_clean[df_clean['Country'] != 'United Kingdom']

print(df_uk.head())

df_uk_model = df_uk[['Sales', 'QuantityInv', 'QuantityRange', 'PriceRange', 'DateRange']]
print(df_uk_model.head())

df_data = df_uk_model.copy()
df_data = pd.get_dummies(df_data, columns=['QuantityRange'], prefix='qr')
df_data = pd.get_dummies(df_data, columns=['PriceRange'], prefix='pr')
df_data = pd.get_dummies(df_data, columns=['DateRange'], prefix='dr')
print(df_data.head())

### Scaling¶
## As the majority of our features are in 0-1 range it would make sense to scale "QuantityInv" feature too.
# In general, scaling features is normally a good idea.

df_data['QuantityInv'] = scale(df_data['QuantityInv'])

### Train-Test Split

#Now we have to split our data into train-test data to be able to train our model and validate its capabilities.

y = df_data['Sales']
X = df_data.drop(columns=['Sales'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)

### Testing and validating different models

## Here we use GridSearch and CrossValidation to test three types of regressors:
# Linear, DecisionTree and RandomForest. This can take a while...

# Linear Regression
fit_intercepts = [True, False]
param_grid_linear = dict(fit_intercept=fit_intercepts)
linear_model = LinearRegression()

# Decision Tree
min_tree_splits = range(2,3)
min_tree_leaves = range(2,3)
param_grid_tree = dict(min_samples_split=min_tree_splits,
                       min_samples_leaf=min_tree_leaves)
tree_model = DecisionTreeRegressor()

# Random Forest
estimators_space = [100]
min_sample_splits = range(1,4)
min_sample_leaves = range(1,3)
param_grid_forest = dict(min_samples_split=min_sample_splits,
                       min_samples_leaf=min_sample_leaves,
                       n_estimators=estimators_space)
forest_model = RandomForestRegressor()

cv = 5

models_to_test = ['LinearRegression','DecisionTreeRegressor','RandomForest']
regression_dict = dict(LinearRegression=linear_model,
                       DecisionTreeRegressor=tree_model,
                       RandomForest=forest_model)
param_grid_dict = dict(LinearRegression=param_grid_linear,
                       DecisionTreeRegressor=param_grid_tree,
                       RandomForest=param_grid_forest)

score_dict = {}
params_dict = {}
mae_dict = {}
mse_dict = {}
r2_dict = {}
best_est_dict = {}

for model in models_to_test:
  regressor = GridSearchCV(regression_dict[model], param_grid_dict[model], cv=cv, n_jobs=-1)

  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)

 # Print the tuned parameters and score
  print(" === Start report for regressor {} ===".format(model))
  score_dict[model] = regressor.best_score_
  print("Tuned Parameters: {}".format(regressor.best_params_))
  params_dict = regressor.best_params_
  print("Best score is {}".format(regressor.best_score_))

  # Compute metrics
  mae_dict[model] = mean_absolute_error(y_test, y_pred)
  print("MAE for {}".format(model))
  print(mean_absolute_error(y_test, y_pred))
  mse_dict[model] = mean_squared_error(y_test, y_pred)
  print("MSE for {}".format(model))
  print(mean_squared_error(y_test, y_pred))
  r2_dict[model] = r2_score(y_test, y_pred)
  print("R2 score for {}".format(model))
  print(r2_score(y_test, y_pred))
  print(" === End of report for regressor {} === \n".format(model))
  
  # Add best estimator to the dict
  best_est_dict[model] = regressor.best_estimator_


