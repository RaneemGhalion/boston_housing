# %% [markdown]
# # Housing Price Prediction Case Study

# %% [markdown]
# ## Multiple Linear Regression
# 
# ### Problem Statement:
# 
# Consider a real estate company that has a dataset containing the prices of properties in the Delhi region. It wishes to use the data to optimise the sale prices of the properties based on important factors such as area, bedrooms, parking, etc.
# 
# Essentially, the company wants —
# 
# 
# - To identify the variables affecting house prices, e.g. area, number of rooms, bathrooms, etc.
# 
# - To create a linear model that quantitatively relates house prices with variables such as number of rooms, area, number of bathrooms, etc.
# 
# - To know the accuracy of the model, i.e. how well these variables can predict house prices.
# 
# ### Data
# Use housing dataset.

# %% [markdown]
# ## Reading and Understanding the Data

# %%
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation

import matplotlib.pyplot as plt 
import seaborn as sns

# %%
housing = pd.DataFrame(pd.read_csv("Housing.csv"))  

# %%
# Check the head of the dataset
housing.head()

# %% [markdown]
# ## Data Inspection

# %%
housing.shape

# %%
housing.info()

# %%
housing.describe()

# %% [markdown]
# ## Data Cleaning

# %%
# Checking Null values
housing.isnull().sum()*100/housing.shape[0]
# There are no NULL values in the dataset, hence it is clean.

# %%
# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(housing['price'], ax = axs[0,0])
plt2 = sns.boxplot(housing['area'], ax = axs[0,1])
plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])
plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])

plt.tight_layout()

# %%
# Outlier Treatment
# Price and area have considerable outliers.
# We can drop the outliers as we have sufficient data.

# %%
# outlier treatment for price
plt.boxplot(housing.price)
Q1 = housing.price.quantile(0.25)
Q3 = housing.price.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.price >= Q1 - 1.5*IQR) & (housing.price <= Q3 + 1.5*IQR)]

# %%
# outlier treatment for area
plt.boxplot(housing.area)
Q1 = housing.area.quantile(0.25)
Q3 = housing.area.quantile(0.75)
IQR = Q3 - Q1
housing = housing[(housing.area >= Q1 - 1.5*IQR) & (housing.area <= Q3 + 1.5*IQR)]

# %%
# Outlier Analysis
fig, axs = plt.subplots(2,3, figsize = (10,5))
plt1 = sns.boxplot(housing['price'], ax = axs[0,0])
plt2 = sns.boxplot(housing['area'], ax = axs[0,1])
plt3 = sns.boxplot(housing['bedrooms'], ax = axs[0,2])
plt1 = sns.boxplot(housing['bathrooms'], ax = axs[1,0])
plt2 = sns.boxplot(housing['stories'], ax = axs[1,1])
plt3 = sns.boxplot(housing['parking'], ax = axs[1,2])

plt.tight_layout()

# %% [markdown]
# ## Exploratory Data Analytics
# 
# Let's now spend some time doing what is arguably the most important step - **understanding the data**.
# - If there is some obvious multicollinearity going on, this is the first place to catch it
# - Here's where you'll also identify if some predictors directly have a strong association with the outcome variable

# %% [markdown]
# ### Visualising Numeric Variables
# 
# Let's make a pairplot of all the numeric variables

# %%
sns.pairplot(housing)
plt.show()

# %% [markdown]
# #### Visualising Categorical Variables
# 
# As you might have noticed, there are a few categorical variables as well. Let's make a boxplot for some of these variables.

# %%
plt.figure(figsize=(20, 12))
plt.subplot(2,3,1)
sns.boxplot(x = 'mainroad', y = 'price', data = housing)
plt.subplot(2,3,2)
sns.boxplot(x = 'guestroom', y = 'price', data = housing)
plt.subplot(2,3,3)
sns.boxplot(x = 'basement', y = 'price', data = housing)
plt.subplot(2,3,4)
sns.boxplot(x = 'hotwaterheating', y = 'price', data = housing)
plt.subplot(2,3,5)
sns.boxplot(x = 'airconditioning', y = 'price', data = housing)
plt.subplot(2,3,6)
sns.boxplot(x = 'furnishingstatus', y = 'price', data = housing)
plt.show()

# %% [markdown]
# We can also visualise some of these categorical features parallely by using the `hue` argument. Below is the plot for `furnishingstatus` with `airconditioning` as the hue.

# %%
plt.figure(figsize = (10, 5))
sns.boxplot(x = 'furnishingstatus', y = 'price', hue = 'airconditioning', data = housing)
plt.show()

# %% [markdown]
# ## Data Preparation

# %% [markdown]
# - You can see that your dataset has many columns with values as 'Yes' or 'No'.
# 
# - But in order to fit a regression line, we would need numerical values and not string. Hence, we need to convert them to 1s and 0s, where 1 is a 'Yes' and 0 is a 'No'.

# %%
# List of variables to map

varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']

# Defining the map function
def binary_map(x):
    return x.map({'yes': 1, "no": 0})

# Applying the function to the housing list
housing[varlist] = housing[varlist].apply(binary_map)

# %%
# Check the housing dataframe now

housing.head()

# %% [markdown]
# ### Dummy Variables

# %% [markdown]
# The variable `furnishingstatus` has three levels. We need to convert these levels into integer as well. 
# 
# For this, we will use something called `dummy variables`.

# %%
# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'
status = pd.get_dummies(housing['furnishingstatus'])

# %%
# Check what the dataset 'status' looks like
status.head()

# %% [markdown]
# Now, you don't need three columns. You can drop the `furnished` column, as the type of furnishing can be identified with just the last two columns where — 
# - `00` will correspond to `furnished`
# - `01` will correspond to `unfurnished`
# - `10` will correspond to `semi-furnished`

# %%
# Let's drop the first column from status df using 'drop_first = True'

status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)

# %%
# Add the results to the original housing dataframe

housing = pd.concat([housing, status], axis = 1)

# %%
# Now let's see the head of our dataframe.

housing.head()

# %%
housing.columns

# %%
# Drop 'furnishingstatus' as we have created the dummies for it

housing.drop(['furnishingstatus'], axis = 1, inplace = True)

# %%
housing.head()

# %% [markdown]
# ### Splitting the Data into Training and Testing Sets

# %%
from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
np.random.seed(0)
df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)

# %% [markdown]
# ### Rescaling the Features 
# 
# As you saw in the demonstration for Simple Linear Regression, scaling doesn't impact your model. Here we can see that except for `area`, all the columns have small integer values. So it is extremely important to rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients. This might become very annoying at the time of model evaluation. So it is advised to use standardization or normalization so that the units of the coefficients obtained are all on the same scale. As you know, there are two common ways of rescaling:
# 
# 1. Min-Max scaling 
# 2. Standardisation (mean-0, sigma-1) 
# 
# This time, we will use MinMax scaling.

# %%
from sklearn.preprocessing import MinMaxScaler

# %%
scaler = MinMaxScaler()

# %%
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']

df_train[num_vars] = scaler.fit_transform(df_train[num_vars])

# %%
import pickle
pickle.dump(scaler,open("scaling.pkl",'wb'))

# %%
df_train.head()

# %%
df_train.describe()

# %%
# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (16, 10))
sns.heatmap(df_train.corr(), annot = True, cmap="YlGnBu")
plt.show()

# %% [markdown]
# As you might have noticed, `area` seems to the correlated to `price` the most. Let's see a pairplot for `area` vs `price`.

# %% [markdown]
# ### Dividing into X and Y sets for the model building

# %%
y_train = df_train.pop('price')
X_train = df_train

# %% [markdown]
# ## Model Building

# %% [markdown]
# This time, we will be using the **LinearRegression function from SciKit Learn** for its compatibility with RFE (which is a utility from sklearn)

# %% [markdown]
# ### RFE

# %% [markdown]
# Recursive feature elimination

# %%
# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# %%
# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

# %%
from sklearn.feature_selection import RFE

rfe = RFE(estimator=lm, n_features_to_select=6)  # Specify the number of features with the keyword
rfe = rfe.fit(X_train, y_train)


# %%
list(zip(X_train.columns,rfe.support_,rfe.ranking_))

# %%
col = X_train.columns[rfe.support_]
col

# %%
X_train.columns[~rfe.support_]

# %% [markdown]
# ### Building model using statsmodel, for the detailed statistics

# %%
# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]

# %%
# Adding a constant variable 
import statsmodels.api as sm  
X_train_rfe = sm.add_constant(X_train_rfe)

# %%
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model

# %%
#Let's see the summary of our linear model
print(lm.summary())

# %%
# Calculate the VIFs for the model
from statsmodels.stats.outliers_influence import variance_inflation_factor

# %%
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# %% [markdown]
# ## Residual Analysis of the train data

# %% [markdown]
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# %%
y_train_price = lm.predict(X_train_rfe)

# %%
res = (y_train_price - y_train)

# %%
# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %%
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label

# %%
plt.scatter(y_train,res)
plt.show()

# %%
# There may be some relation in the error terms.

# %% [markdown]
# ## Model Evaluation

# %% [markdown]
# #### Applying the scaling on the test sets

# %%
num_vars = ['area','stories', 'bathrooms', 'airconditioning', 'prefarea','parking','price']

# %%
df_test[num_vars] = scaler.fit_transform(df_test[num_vars])

# %% [markdown]
# #### Dividing into X_test and y_test

# %%
y_test = df_test.pop('price')
X_test = df_test

# %%
# Adding constant variable to test dataframe
X_test = sm.add_constant(X_test)

# %%
# Now let's use our model to make predictions.

# %%
# Creating X_test_new dataframe by dropping variables from X_test
X_test_rfe = X_test[X_train_rfe.columns]

# %%
# Making predictions
y_pred = lm.predict(X_test_rfe)

# %%
from sklearn.metrics import r2_score 
r2_score(y_test, y_pred)

# %%
# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label

# %% [markdown]
# 
# We can see that the equation of our best fitted line is:
# 
# $ price = 0.35  \times  area + 0.20  \times  bathrooms + 0.19 \times stories+ 0.10 \times airconditioning + 0.10 \times parking + 0.11 \times prefarea $
# 

# %%
import pickle

# %%
pickle.dump(lm,open('regmodel.pkl','wb'))

# %%
pickled_model=pickle.load(open('regmodel.pkl','rb'))

# %%
data=np.array(housing)

# %%


# %%
pickled_model.predict(scaler.transform(data[0].reshape(1,-1)))

# %%



