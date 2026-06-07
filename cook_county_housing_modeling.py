"""Cook County Housing Modeling

A cleaned, GitHub-ready notebook combining exploratory analysis and predictive modeling for residential property data.
"""


# # Cook County Housing Modeling  A cleaned, GitHub-ready notebook combining exploratory analysis and predictive modeling for residential property data.

# ## Exploratory analysis
import numpy as np

import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import zipfile
import os

# Plot settings
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12

with zipfile.ZipFile('data/cook_county_data.zip') as item:
    with item.open("cook_county_train.csv") as f:
        initial_data = pd.read_csv(f, index_col='Unnamed: 0')

# 204,792 observations and 62 features in training data
assert initial_data.shape == (204792, 62)
# Sale Price is provided in the training data
assert 'Sale Price' in initial_data.columns.values

initial_data.columns.values

initial_data['Description'][0]

def plot_distribution(data, label):
    fig, axs = plt.subplots(nrows=2)

    sns.distplot(
        data[label],
        ax=axs[0]
    )
    sns.boxplot(
        x=data[label],
        width=0.3,
        ax=axs[1],
        showfliers=False,
    )

    # Align axes
    spacer = np.max(data[label]) * 0.05
    xmin = np.min(data[label]) - spacer
    xmax = np.max(data[label]) + spacer
    axs[0].set_xlim((xmin, xmax))
    axs[1].set_xlim((xmin, xmax))

    # Remove some axis text
    axs[0].xaxis.set_visible(False)
    axs[0].yaxis.set_visible(False)
    axs[1].yaxis.set_visible(False)

    # Put the two plots together
    plt.subplots_adjust(hspace=0)
    fig.suptitle("Distribution of " + label)

plot_distribution(initial_data, label='Sale Price')

no_right_outliers = initial_data[initial_data['Sale Price'] <= 1000000]['Sale Price']
plt.hist(no_right_outliers, bins=200)
plt.xlabel("Sale Price")
plt.ylabel("Count")
plt.title("Distribution of Sale Price")
plt.show()

initial_data['Sale Price'].value_counts()

counts = initial_data['Sale Price'].value_counts().reset_index()
low_prices = counts[(counts['Sale Price'] < 500) & (counts['Sale Price'] >= 0)]
q2a = low_prices['Sale Price'].head(4).tolist()
q2a

len(initial_data[initial_data['Sale Price'] > 1000000]) / len(initial_data[initial_data['Sale Price'] >= 500])

df = initial_data.copy()
training_data = df[df['Sale Price'] >= 500]
training_data['Log Sale Price'] = np.log(training_data['Sale Price'])
training_data

plot_distribution(training_data, label='Log Sale Price');

training_data['Log Building Square Feet'] = np.log(training_data['Building Square Feet'])

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    """
    Input:
      data (DataFrame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than or equal to this will be removed
      upper (numeric): observations with values higher than or equal to this will be removed

    Output:
      a DataFrame with outliers removed

    Note: This function should not change or mutate the contents of data.
    """
    copy = data.copy()
    copy = copy[(copy[variable] < upper) & (copy[variable] > lower)]
    return copy

# Run this to see the distribution!
plot_distribution(training_data, label='Estimate (Land)');

# This should be set to True or False
q4bstatement = True

lower = training_data['Estimate (Land)'].quantile(0.25)
upper = training_data['Estimate (Land)'].quantile(0.75)
IQR = upper - lower

q4c_training_data = remove_outliers(training_data, 'Estimate (Land)', lower - IQR * 1.5, upper + 1.5 * IQR)

plot_distribution(q4c_training_data, label='Estimate (Land)');

# optional cell for scratch work

q5a = [1, 2, 4, 6, 7, 8]

# optional cell for scratch work

training_data['Description'][1]

def add_total_bathrooms(data):
    """
    Input:
      data (DataFrame): a DataFrame containing at least the Description column.

    Output:
      a Dataframe with a new column "Bathrooms" containing floats.

    """
    with_rooms = data.copy()
    with_rooms['Bathrooms'] = with_rooms['Description'].str.extract(r"bedrooms, and ([\d.]+) of which are bathrooms")
    with_rooms["Bathrooms"] = pd.to_numeric(with_rooms["Bathrooms"], errors="coerce").fillna(0.0)

    return with_rooms

training_data = add_total_bathrooms(training_data)

main = data[data["Bathrooms"] < 20]
outlier = data[data["Bathrooms"] >= 20]

plt.plot(main["Bathrooms"], main["Log Sale Price"], marker="o", color="blue")
plt.scatter(outlier["Bathrooms"], outlier["Log Sale Price"], color="red", marker="x", s=80, label="Outlier")
plt.title("Median Log Sale Price vs. Number of Bathrooms", fontsize=14)
plt.xlabel("Number of Bathrooms (including half baths)", fontsize=12)
plt.ylabel("Median Log Sale Price ($)", fontsize=12)
plt.xticks(range(0, int(data["Bathrooms"].max())+2, 2))
plt.legend();

num_neighborhoods = training_data['Neighborhood Code'].unique().shape[0]
num_neighborhoods

# Feel free to create a cell below this and run plot_categorical(training_data) if you want to see what this function outputs.
def plot_categorical(neighborhoods):
    fig, axs = plt.subplots(nrows=2)

    sns.boxplot(
        x='Neighborhood Code',
        y='Log Sale Price',
        data=neighborhoods,
        ax=axs[0],
    )

    sns.countplot(
        x='Neighborhood Code',
        data=neighborhoods,
        ax=axs[1],
    )

    # Draw median price
    axs[0].axhline(
        y=training_data['Log Sale Price'].median(),
        color='red',
        linestyle='dotted'
    )

    # Label the bars with counts
    for patch in axs[1].patches:
        x = patch.get_bbox().get_points()[:, 0]
        y = patch.get_bbox().get_points()[1, 1]
        axs[1].annotate(f'{int(y)}', (x.mean(), y), ha='center', va='bottom')

    # Format x-axes
    axs[1].set_xticklabels(axs[1].xaxis.get_majorticklabels(), rotation=90)
    axs[0].xaxis.set_visible(False)

    # Narrow the gap between the plots
    plt.subplots_adjust(hspace=0.01)

copy = training_data.copy()
df = copy['Neighborhood Code'].value_counts().reset_index()
top_10_neighborhood_codes = df['Neighborhood Code'].iloc[:10]
in_top_10_neighborhoods = copy[copy['Neighborhood Code'].isin(top_10_neighborhood_codes)]

plot_categorical(neighborhoods=in_top_10_neighborhoods)

def find_expensive_neighborhoods(data, n=3, metric=np.median):
    """
    Input:
      data (DataFrame): should contain at least an int-valued 'Neighborhood Code'
        and a numeric 'Log Sale Price' column
      n (int): the number of top values desired
      metric (function): function used for aggregating the data in each neighborhood.
        for example, np.median for median prices

    Output:
      a list of the the neighborhood codes of the top n highest-priced neighborhoods
      as measured by the metric function
    """
    codes = data['Neighborhood Code']
    prices = data['Log Sale Price']

    grouped = data.groupby('Neighborhood Code')['Log Sale Price'].agg(metric).reset_index()

    neighborhoods = grouped.sort_values('Log Sale Price', ascending = False).iloc[:n, 0]

    # This makes sure the final list contains the generic int type used in Python3, not specific ones used in NumPy.
    return [int(code) for code in neighborhoods]

expensive_neighborhoods = find_expensive_neighborhoods(training_data, 3, np.median)
expensive_neighborhoods

def add_in_expensive_neighborhood(data, expensive_neighborhoods):
    """
    Input:
      data (DataFrame): a DataFrame containing a 'Neighborhood Code' column with values
        found in the codebook
      expensive_neighborhoods (list of ints): ints should be the neighborhood codes of
        neighborhoods pre-identified as expensive
    Output:
      DataFrame identical to the input with the addition of a binary
      in_expensive_neighborhood column
    """
    data['in_expensive_neighborhood'] = data['Neighborhood Code'].isin(expensive_neighborhoods).astype(int)
    return data

expensive_neighborhoods = find_expensive_neighborhoods(training_data, 3, np.median)
training_data = add_in_expensive_neighborhood(training_data, expensive_neighborhoods)

def substitute_wall_material(data):
    """
    Input:
      data (DataFrame): a DataFrame containing a 'Wall Material' column.  Its values
                         should be limited to those found in the codebook
    Output:
      new DataFrame identical to the input except with a refactored 'Wall Material' column
    """
    new_data = data.copy()
    new_data['Wall Material'] = new_data['Wall Material'].replace({1: "Wood", 2: "Masonry", 3: "Wood&Masonry", 4: "Stucco"})
    return new_data

training_data_mapped = substitute_wall_material(training_data)
training_data_mapped.head()

from sklearn.preprocessing import OneHotEncoder

def ohe_wall_material(data):
    """
    One-hot-encodes wall material. New columns are of the form "Wall Material_MATERIAL".
    """
    data = data.copy()
    cat = ['Wall Material']

    oh_enc = OneHotEncoder()
    oh_enc.fit(data[cat])

    cat_data = oh_enc.transform(data[cat]).toarray()
    cat_df = pd.DataFrame(data = cat_data, columns = oh_enc.get_feature_names_out(), index = data.index)
    return data.join(cat_df)

training_data_ohe = ohe_wall_material(training_data_mapped)
# This line of code will display only the one-hot-encoded columns in training_data_ohe that
# have names that begin with “Wall Material_"
training_data_ohe.filter(regex='^Wall Material_').head(10)


# ## Model development
import numpy as np

import pandas as pd
from pandas.api.types import CategoricalDtype

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model as lm

import warnings
warnings.filterwarnings("ignore")

import zipfile
import os

from ds100_utils import *
from feature_func import *

# Plot settings
plt.rcParams['figure.figsize'] = (12, 9)
plt.rcParams['font.size'] = 12

with zipfile.ZipFile('cook_county_data.zip') as item:
    item.extractall()

training_val_data = pd.read_csv("cook_county_train_val.csv", index_col='Unnamed: 0')
test_data = pd.read_csv("cook_county_contest_test.csv", index_col='Unnamed: 0')

# 204792 observations and 62 features in training data
assert training_val_data.shape == (204792, 62)

# 55311 observations and 61 features in test data
assert test_data.shape == (55311, 61)

# Sale Price is provided in the training/validation data
assert 'Sale Price' in training_val_data.columns.values

# Sale Price and Log Sale Price is hidden in the test data
assert 'Sale Price' not in test_data.columns.values
assert 'Log Sale Price' not in test_data.columns.values

training_val_data.columns.values

q1c = "A"

# This makes the train-validation split in this section reproducible across different runs
# of the notebook. You do not need this line to run train_val_split in general.

# DO NOT CHANGE THIS LINE
np.random.seed(1337)
# DO NOT CHANGE THIS LINE

def train_val_split(data):
    """
    Takes in a DataFrame `data` and randomly splits it into two smaller DataFrames
    named `train` and `validation` with 80% and 20% of the data, respectively.
    """

    data_len = data.shape[0]
    shuffled_indices = np.random.permutation(data_len)

    train_size = int(0.8 * data_len)
    train_indices = shuffled_indices[:train_size]
    validation_indices = shuffled_indices[train_size:]

    train = data.iloc[train_indices]
    validation = data.iloc[validation_indices]

    return train, validation
train, validation = train_val_split(training_val_data)

q3a = ">="


###### Copy any function you would like to below ######
...
#######################################################

def feature_engine_simple(data):
    # Remove outliers
    data = remove_outliers(data, 'Sale Price', lower=499)
    # Create Log Sale Price column
    data = log_transform(data, 'Sale Price')
    # Create Bathrooms column
    data = add_total_bathrooms (data)
    # Select X and Y from the full data
    X = data[['Bathrooms']]
    Y = data['Log Sale Price']
    return X, Y

# Reload the data
full_data = pd.read_csv("cook_county_train.csv")

# Process the data using the pipeline for the first model.
np.random.seed(1337)
train_m1, valid_m1 = train_val_split(full_data)
X_train_m1_simple, Y_train_m1_simple = feature_engine_simple(train_m1)
X_valid_m1_simple, Y_valid_m1_simple = feature_engine_simple(valid_m1)

# Take a look at the result
display(X_train_m1_simple.head())
display(Y_train_m1_simple.head())

# Run this cell to define feature_engine_pipe; no further action is needed.
def feature_engine_pipe(data, pipeline_functions, prediction_col):
    """Process the data for a guided model."""
    for function, arguments, keyword_arguments in pipeline_functions:
        if keyword_arguments and (not arguments):
            data = data.pipe(function, **keyword_arguments)
        elif (not keyword_arguments) and (arguments):
            data = data.pipe(function, *arguments)
        else:
            data = data.pipe(function)
    X = data.drop(columns=[prediction_col])
    Y = data.loc[:, prediction_col]
    return X, Y

# Reload the data
full_data = pd.read_csv("cook_county_train.csv")

# Apply feature engineering to the data using the pipeline for the first model
np.random.seed(1337)
train_m1, valid_m1 = train_val_split(full_data)

# Helper function
def select_columns(data, *columns):
    """Select only columns passed as arguments."""
    return data.loc[:, columns]

# Pipelines, a list of tuples
m1_pipelines = [
    (remove_outliers, None, {
        'variable': 'Sale Price',
        'lower': 499,
    }),
    (log_transform, None, {'col': 'Sale Price'}),
    (add_total_bathrooms, None, None),
    (select_columns, ['Log Sale Price', 'Bathrooms'], None)
]

X_train_m1, Y_train_m1 = feature_engine_pipe(train_m1, m1_pipelines, 'Log Sale Price')
X_valid_m1, Y_valid_m1 = feature_engine_pipe(valid_m1, m1_pipelines, 'Log Sale Price')

# Take a look at the result
# It should be the same above as the result returned by feature_engine_simple
display(X_train_m1.head())
display(Y_train_m1.head())

# DO NOT CHANGE THIS LINE
np.random.seed(1337)
# DO NOT CHANGE THIS LINE

# Process the data using the pipeline for the second model
train_m2, valid_m2 = train_val_split(full_data)

m2_pipelines = [
    (remove_outliers, None, {
            'variable': 'Sale Price',
            'lower': 499,
        }),
    (log_transform, None, {'col': 'Sale Price'}),
    (log_transform, None, {'col': 'Building Square Feet'}),
    (add_total_bathrooms, None, None),
    (select_columns, ['Log Sale Price', 'Bathrooms', 'Log Building Square Feet'], None)

]

X_train_m2, Y_train_m2 = feature_engine_pipe(train_m2, m2_pipelines, 'Log Sale Price')
X_valid_m2, Y_valid_m2 =feature_engine_pipe(valid_m2, m2_pipelines, 'Log Sale Price')

# Take a look at the result
display(X_train_m2.head())
display(Y_train_m2.head())

linear_model_m1 = lm.LinearRegression(fit_intercept=True)
linear_model_m2 = lm.LinearRegression(fit_intercept=True)

# Fit the 1st model
linear_model_m1.fit(X_train_m1, Y_train_m1)
# Compute the fitted and predicted values of Log Sale Price for 1st model
Y_fitted_m1 = linear_model_m1.predict(X_train_m1)
Y_predicted_m1 = linear_model_m1.predict(X_valid_m1)

# Fit the 2nd model
linear_model_m2.fit(X_train_m2, Y_train_m2)
# Compute the fitted and predicted values of Log Sale Price for 2nd model
Y_fitted_m2 = linear_model_m2.predict(X_train_m2)
Y_predicted_m2 = linear_model_m2.predict(X_valid_m2)

def rmse(predicted, actual):
    """
    Calculates RMSE from actual and predicted values.
    Input:
      predicted (1D array): Vector of predicted/fitted values
      actual (1D array): Vector of actual values
    Output:
      A float, the RMSE value.
    """
    return np.sqrt(np.mean((actual - predicted)**2))

residuals_m2 = Y_valid_m2 - Y_predicted_m2
plt.scatter(Y_valid_m2, residuals_m2, s = 5, alpha = 0.2)
plt.title("Log Sale Price Value vs. Model 2 Residuals")
plt.xlabel("Log Sale Price Values")
plt.ylabel("Residuals (y - y_predicted) ");

q4b = "regressive"

from sklearn.model_selection import KFold

def compute_CV_error(X_train, Y_train, folds=10):
    """
    Split the training data into `k` subsets.
    For each subset,
        - Fit a model holding out that subset.
        - Compute the MSE on that subset (the validation set).
    You should be fitting `k` models in total.
    Return a list of `k` RMSEs.

    Args:
        model: An sklearn model with fit and predict functions.
        X_train (DataFrame): Training data.
        Y_train (Series): Label.

    Return:
         A list of `k` RMSEs.
    """
    kf = KFold(n_splits=folds)
    validation_errors = []

    for train_idx, valid_idx in kf.split(X_train):
        # Split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        split_Y_train, split_Y_valid = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]

        # Fit the model on the training split
        model.fit(split_X_train, split_Y_train)

        # Compute the RMSE on the validation split
        error = rmse(split_Y_valid, model.predict(split_X_valid))

        validation_errors.append(error)

    return validation_errors

# DO NOT CHANGE THIS LINE
np.random.seed(1337)

# MODIFY THESE LINES
cv_m1 = compute_CV_error(X_train_m1, Y_train_m1, 4)
cv_m2 = compute_CV_error(X_train_m2, Y_train_m2, 4)
print(f"The RMSE errors for 4-fold cross-validation on Model 1 were: {cv_m1}")
print(f"The RMSE errors for 4-fold cross-validation on Model 2 were: {cv_m2}")

#loading datasets
original_training_data = pd.read_csv("cook_county_train_val.csv", index_col='Unnamed: 0')
original_est_data = pd.read_csv("cook_county_contest_test.csv", index_col='Unnamed: 0')

original_training_data.columns

# Add any EDA code below

x = original_training_data["Repair Condition"]
y = np.log(original_training_data[original_training_data['Sale Price'] >= 500].loc[:, "Sale Price"])
sns.scatterplot(data = q4c_training_data, x = x, y = y)

# Define any additional helper functions or variables you need here
def add_total_bathrooms(data):
    """
    Input:
      data (DataFrame): a DataFrame containing at least the Description column.

    Output:
      a Dataframe with a new column "Bathrooms" containing floats.

    """
    with_rooms = data.copy()
    with_rooms['Bathrooms'] = with_rooms['Description'].str.extract(r"bedrooms, and ([\d.]+) of which are bathrooms")
    with_rooms["Bathrooms"] = pd.to_numeric(with_rooms["Bathrooms"], errors="coerce").fillna(0.0)

    return with_rooms

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    """
    Input:
      data (DataFrame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than or equal to this will be removed
      upper (numeric): observations with values higher than or equal to this will be removed

    Output:
      a DataFrame with outliers removed

    Note: This function should not change or mutate the contents of data.
    """
    copy = data.copy()
    copy = copy[(copy[variable] < upper) & (copy[variable] > lower)]
    return copy

def train_val_split(data):
    """
    Takes in a DataFrame `data` and randomly splits it into two smaller DataFrames
    named `train` and `validation` with 80% and 20% of the data, respectively.
    """

    data_len = data.shape[0]
    shuffled_indices = np.random.permutation(data_len)

    train_size = int(0.8 * data_len)
    train_indices = shuffled_indices[:train_size]
    validation_indices = shuffled_indices[train_size:]

    train = data.iloc[train_indices]
    validation = data.iloc[validation_indices]

    return train, validation

# Please include all of your feature engineering processes inside this function.
# Do not modify the parameters of this function.
def feature_engine_final(data, is_test_set=False):
    # Whenever you access 'Log Sale Price' or 'Sale Price', make sure to use the
    # condition is_test_set like this:
    if not is_test_set:
        # Processing for the training set (i.e. not the test set)
        # CAN involve references to sale price!
        # CAN involve filtering certain rows or removing outliers

        #Editing predicted values to make more sense, introducing log transformation to open up data
        #1) removed properties with sale prices below 500
        data = data[data['Sale Price'] >= 500]

        #2) Adding Log Sale Price column
        data['Log Sale Price'] = np.log(data['Sale Price'])

        #Feature 1: Land Estimates, removed outliers using IQR definition
        lower = data['Estimate (Land)'].quantile(0.25)
        upper = data['Estimate (Land)'].quantile(0.75)
        IQR = upper - lower
        data = remove_outliers(data, 'Estimate (Land)', lower - IQR * 1.5, upper + 1.5 * IQR)

        #Feature 2: Adding Log Building Square Feet Column
        data['Log Building Square Feet'] = np.log(data['Building Square Feet']) #adding building square feet column

        #Feature 3: Adding Bathrooms Column, then removing properties with more than 20 bathrooms
        data = add_total_bathrooms(data)
        data = data[data["Bathrooms"] < 20]

    # Processing for both test and training set
    # CANNOT involve references to sale price!
    # CANNOT involve removing any rows

    # create engineered columns here
    data['Log_Bldg_SF']   = np.log1p(data['Building Square Feet'])
    data['Log_Land_SF']   = np.log1p(data['Land Square Feet'].fillna(0))
    data['Log_Lot_Size']  = np.log1p(data['Lot Size'].fillna(0))
    data['Log_Est_Land']  = np.log1p(data['Estimate (Land)'].clip(lower=0))
    data['Log_Est_Bldg']  = np.log1p(data['Estimate (Building)'].clip(lower=0))
    data = add_total_bathrooms(data)

    # ratios
    data['FAR'] = data['Building Square Feet'] / data['Lot Size'].replace(0, np.nan)
    far_hi = np.nanpercentile(data['FAR'].dropna(), 99)
    data['FAR'] = data['FAR'].clip(0, far_hi).fillna(0)

    data['Garage_Area_per_SF'] = (
        data['Garage 1 Area'].fillna(0) + data['Garage 2 Area'].fillna(0)
    ) / data['Building Square Feet'].replace(0, np.nan)
    data['Garage_Area_per_SF'] = data['Garage_Area_per_SF'].clip(
        0, np.nanpercentile(data['Garage_Area_per_SF'].dropna(), 99)
    ).fillna(0)

    # binaries (converting yes or no variables to 0 or 1)
    data['Has_Central_Air'] = (data['Central Air'] == 1).astype(int)
    data['Has_Fireplace']   = (data['Fireplaces'] > 0).astype(int)

    # time controls (changing column names to match my custom formatting)
    data['Sale_Year']  = data['Sale Year']
    data['Sale_Qtr']   = data['Sale Quarter']
    data['Sale_Month'] = data['Sale Month of Year']

    #my feature pack
    FEATURES = [
        'Log_Bldg_SF','Log_Land_SF','Log_Lot_Size','FAR',
        'Repair Condition','Garage 1 Size','Garage_Area_per_SF',
        'Has_Central_Air','Has_Fireplace',
        'Log_Est_Land','Log_Est_Bldg',
        'Sale_Year','Sale_Qtr','Sale_Month', 'Bathrooms'
    ]

    # Return predictors (X) and response (Y) variables separately
    if not is_test_set:
        # Predictors. Your X should not include Log Sale Price!
        X = data.loc[:, [c for c in FEATURES if c in data.columns]]
        Y = data.loc[:, "Log Sale Price"]

        return X, Y

    else:
        # Predictors
        X = data.loc[:, [c for c in FEATURES if c in data.columns]]
        return X

# DO NOT EDIT THESE TWO LINES!
check_rmse_threshold = run_linear_regression_test_optim(lm.LinearRegression(fit_intercept=True), feature_engine_final, 'cook_county_train.csv', None, False)
print("Current training RMSE:", check_rmse_threshold.loss)

def rmse(predicted, actual):
    """
    Calculates RMSE from actual and predicted values.
    Input:
      predicted (1D array): Vector of predicted/fitted values
      actual (1D array): Vector of actual values
    Output:
      A float, the RMSE value.
    """
    return np.sqrt(np.mean((actual - predicted)**2))

# Use this space to evaluate your model
# if you reset your memory, you need to define the functions again
import sklearn.linear_model as lm

from sklearn.model_selection import KFold

def compute_CV_error_2(X_train, Y_train, folds=10):
    """
    Split the training data into `k` subsets.
    For each subset,
        - Fit a model holding out that subset.
        - Compute the MSE on that subset (the validation set).
    You should be fitting `k` models in total.
    Return a list of `k` RMSEs.

    Args:
        model: An sklearn model with fit and predict functions.
        X_train (DataFrame): Training data.
        Y_train (Series): Label.

    Return:
         A list of `k` RMSEs.
    """
    kf = KFold(n_splits=folds)
    validation_errors = []

    for train_idx, valid_idx in kf.split(X_train):
        # Split the data
        split_X_train, split_X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        split_Y_train, split_Y_valid = Y_train.iloc[train_idx], Y_train.iloc[valid_idx]

        # Fit the model on the training split
        model.fit(split_X_train, split_Y_train)

        # Compute the RMSE on the validation split
        error = rmse(split_Y_valid, model.predict(split_X_valid))

        validation_errors.append(error)

    return np.mean(validation_errors)

train, test = train_val_split(original_training_data)

X_train, Y_train = feature_engine_final(train, is_test_set=False)

folds = 4
mean_cv_error = compute_CV_error_2(X_train, Y_train, folds = folds)
print(f"Mean {folds} Fold CV Error: {mean_cv_error}")

from datetime import datetime
from IPython.display import display, HTML

Y_test_pred = run_linear_regression_test(lm.LinearRegression(fit_intercept=True), feature_engine_final, None, 'cook_county_train.csv', 'cook_county_contest_test.csv',
                                         is_test = True, is_ranking = False, return_predictions = True
                                         )

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": pd.read_csv('cook_county_contest_test.csv')['Unnamed: 0'],
    "Value": Y_test_pred,
}, columns=['Id', 'Value'])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = "submission_{}.csv".format(timestamp)
submission_df.to_csv(filename, index=False)

#print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
display(HTML("Download your test prediction <a href='" + filename + "' download>here</a>."))

# Scratch space to check if your prediction is reasonable. See 5e for hints.
submission_df["Value"].describe()

# Run the cell below; no further action is needed
train_df = pd.read_csv('cook_county_train.csv')
X, Y_true = feature_engine_final(train_df)
model = lm.LinearRegression(fit_intercept=True)
model.fit(X, Y_true)
Y_pred = model.predict(X)

preds_df = pd.DataFrame({'True Log Sale Price' : Y_true, 'Predicted Log Sale Price' : Y_pred,
                         'True Sale Price' : np.e**Y_true, 'Predicted Sale Price' : np.e**Y_pred})
preds_df.head()

# Run the cell below to obtain the two subsets of data; no further action is needed.
min_Y_true, max_Y_true = np.round(np.min(Y_true), 1) , np.round(np.max(Y_true), 1)
median_Y_true = np.round(np.median(Y_true), 1)
cheap_df = preds_df[(preds_df['True Log Sale Price'] >= min_Y_true) & (preds_df['True Log Sale Price'] <= median_Y_true)]
expensive_df = preds_df[(preds_df['True Log Sale Price'] > median_Y_true) & (preds_df['True Log Sale Price'] <= max_Y_true)]

print(f'\nThe lower interval contains houses with true sale price ${np.round(np.e**min_Y_true)} to ${np.round(np.e**median_Y_true)}')
print(f'The higher interval contains houses with true sale price ${np.round(np.e**median_Y_true)} to ${np.round(np.e**max_Y_true)}\n')

rmse_cheap = rmse(cheap_df['Predicted Sale Price'], cheap_df['True Sale Price'])
rmse_expensive = rmse(expensive_df['Predicted Sale Price'], expensive_df['True Sale Price'])

prop_overest_cheap = np.mean(cheap_df['Predicted Log Sale Price'] > cheap_df['True Log Sale Price'])
prop_overest_expensive = np.mean(expensive_df['Predicted Log Sale Price'] > expensive_df['True Log Sale Price'])

print(f"The RMSE for properties with log sale prices in the interval {(min_Y_true, median_Y_true)} is {np.round(rmse_cheap)}")
print(f"The RMSE for properties with log sale prices in the interval {(median_Y_true, max_Y_true)} is {np.round(rmse_expensive)}\n")
print(f"The percentage of overestimated values for properties with log sale prices in the interval {(min_Y_true, median_Y_true)} is {np.round(100 * prop_overest_cheap, 2)}%")
print(f"The percentage of overestimated values for properties with log sale prices in the interval {(median_Y_true, max_Y_true)} is {np.round(100 * prop_overest_expensive, 2)}%")

def rmse_interval(df, start, end):
    """
    Given a design matrix X and response vector Y, computes the RMSE for a subset of values
    wherein the corresponding Log Sale Price lies in the interval [start, end].

    Input:
    df : pandas DataFrame with columns 'True Log Sale Price',
        'Predicted Log Sale Price', 'True Sale Price', 'Predicted Sale Price'
    start : A float specifying the start of the interval (inclusive)
    end : A float specifying the end of the interval (inclusive)
    """

    subset_df = df[(df['True Log Sale Price'] >= start) & (df['True Log Sale Price'] <= end)]

    rmse_subset = rmse(subset_df['Predicted Sale Price'], subset_df['True Sale Price'])
    return rmse_subset

def prop_overest_interval(df, start, end):
    """
    Given a DataFrame df, computes prop_overest for a subset of values
    wherein the corresponding Log Sale Price lies in the interval [start, end].

    Input:
    df : pandas DataFrame with columns 'True Log Sale Price',
        'Predicted Log Sale Price', 'True Sale Price', 'Predicted Sale Price'
    start : A float specifying the start of the interval (inclusive)
    end : A float specifying the end of the interval (inclusive)
    """

    subset_df = df[(df['True Log Sale Price'] >= start) & (df['True Log Sale Price'] <= end)]

    # DO NOT MODIFY THESE TWO LINES
    if subset_df.shape[0] == 0:
        return -1

    prop_subset = np.mean(subset_df['Predicted Log Sale Price'] > subset_df['True Log Sale Price'])
    return prop_subset

# RMSE plot
plt.figure(figsize = (8,5))
plt.subplot(1, 2, 1)
rmses = []
for i in np.arange(8, 14, 0.5):
    rmses.append(rmse_interval(preds_df, i, i + 0.5))
plt.bar(x = np.arange(8.25, 14.25, 0.5), height = rmses, edgecolor = 'black', width = 0.5)
plt.title('RMSE of Sale Price For Different Intervals\n of Log Sale Price', fontsize = 10)
plt.xlabel('Log Sale Price')
plt.yticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel('RMSE')

# Overestimation plot
plt.subplot(1, 2, 2)
props = []
for i in np.arange(8, 14, 0.5):
    props.append(prop_overest_interval(preds_df, i, i + 0.5) * 100)
plt.bar(x = np.arange(8.25, 14.25, 0.5), height = props, edgecolor = 'black', width = 0.5)
plt.title('Percentage of House Values Overestimated \n for different intervals of Log Sale Price', fontsize = 10)
plt.xlabel('Log Sale Price')
plt.yticks(fontsize = 10)
plt.xticks(fontsize = 10)
plt.ylabel('Percentage of House Values\n that were Overestimated (%)')

plt.tight_layout()
plt.show()

data = pd.read_csv("cook_county_train.csv", index_col='Unnamed: 0')
trainX, trainY = feature_engine_final(data)

# X is the design matrix (including bias column), y is the vector of true outputs, theta is the parameter vector
def mape(theta, X, y):
    y_pred = X @ theta  # compute predicted values using linear combination of features
    percentage_error = np.abs((y - y_pred) / y)  # calculate element-wise percentage errors
    return np.mean(percentage_error)  # return the MAPE

from scipy.optimize import minimize

# Add bias (intercept) column to the design matrix
trainX_with_bias = np.column_stack([np.ones(trainX.shape[0]), trainX])

# Initialize parameter vector with zeros
theta_0 = np.zeros(trainX_with_bias.shape[1])

# Use scipy's minimize to find weights that minimize MAPE
res = minimize(mape, theta_0, args=(trainX_with_bias, trainY), method='BFGS')

# Optimal weights after training
theta_opt = res.x

theta_opt

new_preds_df = pd.DataFrame({
    'True Log Sale Price'     : trainY,
    'Predicted Log Sale Price': trainX_with_bias @ theta_opt,
    'True Sale Price'         : np.e ** trainY,
    'Predicted Sale Price'    : np.e ** (trainX_with_bias @ theta_opt)
})

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)

mape_values = []
for i in np.arange(8, 14, 0.5):
    mape_values.append(mape_interval(new_preds_df, i, i + 0.5))

plt.bar(x=np.arange(8.25, 14.25, 0.5), height=mape_values, edgecolor='black', width=0.5)
plt.title('MAPE of Sale Price Across\n Log Sale Price Intervals', fontsize=10)
plt.xlabel('Log Sale Price')
plt.ylabel('Mean Absolute Percentage Error (MAPE)')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10);

def your_custom_error_metric(theta, X, y): # feel free to include more parameters
    ...

def your_custom_error_metric_interval(df, start, end):
    ...

from scipy.optimize import minimize

trainX_with_bias = np.column_stack([np.ones(trainX.shape[0]), trainX])
theta_0 = np.zeros(trainX_with_bias.shape[1])
# if you adjusted the amount of parameters above, you would have to modify the args parameter as well
res = minimize(your_custom_error_metric, theta_0, args=(trainX_with_bias, trainY), method='BFGS')
theta_opt = res.x

new_preds_df = pd.DataFrame({
    'True Log Sale Price'     : trainY,
    'Predicted Log Sale Price': trainX_with_bias @ theta_opt,
    'True Sale Price'         : np.e ** trainY,
    'Predicted Sale Price'    : np.e ** (trainX_with_bias @ theta_opt)
})

plt.figure(figsize=(8, 5))
plt.subplot(1, 2, 1)

your_values = []
for i in np.arange(8, 14, 0.5):
    your_values.append(your_custom_error_metric_interval(new_preds_df, i, i + 0.5))

plt.bar(x=np.arange(8.25, 14.25, 0.5), height=your_values, edgecolor='black', width=0.5)
plt.title('Custom Error Metric of Sale Price Across\n Log Sale Price Intervals', fontsize=10)
plt.xlabel('Log Sale Price')
plt.ylabel('Custom Error Metric')
plt.xticks(fontsize=10)
plt.yticks(fontsize=10);

