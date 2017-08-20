#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','other', 'from_messages', 'from_this_person_to_poi', 'to_messages']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

my_dataset = data_dict

## Load the data dictionary into a dataframe
data = pd.DataFrame.from_dict(my_dataset,orient='index')



## Replace NA values with zeros
data = data.replace('NaN',np.nan)
data = data.replace(np.nan,0.0)

## Replace negative values with zeros
num = data._get_numeric_data()
num[num < 0] = 0

### Task 2: Remove outliers

## Removing the 'TOTAL' row
data = data.drop('TOTAL')

## Removing the 'THE TRAVEL AGENCY IN THE PARK' since it isn't an individual
data = data.drop('THE TRAVEL AGENCY IN THE PARK')

# Used the outlierCleaner from earlier exercises to remove 15% of point with largest residual error
def outlierCleaner(preds, feature_vals):

    from scipy.stats import percentileofscore

    cleaned_data = []

    diffs = (np.array(feature_vals) - np.array(preds)) ** 2
    diffs = diffs.flatten()
    feature_vals = feature_vals.flatten()

    data = zip(feature_vals, diffs)

    percentiles = [percentileofscore(diffs, i) for i in diffs]

    cleaned_data = [ val if percentileofscore(diffs, diff) < 85 else 0.0 for (val,diff) in data]

    return cleaned_data

rel_cols = ['bonus','deferral_payments',
'deferred_income',
 'director_fees',
 'exercised_stock_options',
 'expenses',
 'from_messages',
 'from_poi_to_this_person',
 'from_this_person_to_poi',
 'loan_advances',
 'long_term_incentive',
 'other',
 'restricted_stock',
 'restricted_stock_deferred',
 'salary',
 'shared_receipt_with_poi',
 'to_messages',
 'total_payments',
 'total_stock_value']



from sklearn.linear_model import LinearRegression

for f in rel_cols:
    curr_feat = np.reshape( np.array(data[f]), (len(data[f]), 1))
    poi = np.reshape( np.array(data.poi), (len(data.poi), 1))

    reg = LinearRegression()
    reg.fit(curr_feat, poi)

    pred = reg.predict(curr_feat)

    cleaned_data = outlierCleaner(pred,curr_feat)

    data[f] = cleaned_data


### Task 3: Create new feature(s)
fin_cols = ['bonus',
            'deferral_payments',
            'deferred_income',
            'director_fees',
            'exercised_stock_options',
            'expenses',
            'loan_advances',
            'long_term_incentive',
            'restricted_stock',
            'restricted_stock_deferred',
            'salary',
            'total_payments',
            'total_stock_value']

### Use PCA to create new featuress
from sklearn.decomposition import PCA

pca = PCA(n_components = 2,svd_solver='randomized')
pca.fit(data[fin_cols])
pca_values = pca.transform(data[fin_cols])
pca_df = pd.DataFrame(pca_values,
                      index = data.index)

data = data.merge(pca_df,left_index=True,right_index=True)
new_cols = data.columns.difference(['poi','email_address'])

# ### Intelligently select features
from sklearn.feature_selection import SelectPercentile, f_classif

selector = SelectPercentile(f_classif,percentile = 20)
selector.fit(data[new_cols],data['poi'])
features_transformed = selector.transform(data[new_cols])

# ### Properly scale features
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
rescaled_weight = scaler.fit_transform(features_transformed)

### Store to my_dataset for easy export below.
my_dataset = data['poi'].reset_index()[['poi']].merge(pd.DataFrame(rescaled_weight, columns= ['other', 'from_messages', 'from_this_person_to_poi', 'to_messages']),left_index=True,right_index=True).to_dict('index')

### Extract features and labels from dataset for local testing



feat_data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(feat_data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# ### Pick an algorithm

from sklearn import model_selection

classifiers = []

features_train, features_test, labels_train, labels_test = model_selection.train_test_split(rescaled_weight,data['poi'], test_size=0.2, random_state=67)


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)

classifiers.append(clf)

clf.score(features_test,labels_test)


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(features_train,labels_train)

classifiers.append(clf)

clf.score(features_test,labels_test,)


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(features_train,labels_train)

classifiers.append(clf)

clf.score(features_test,labels_test)



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': [1,5],
             'algorithm':['auto','ball_tree', 'kd_tree'],
             'n_jobs':[1,5],
             'weights':['uniform','distance']}

clf = GridSearchCV(KNeighborsClassifier(), param_grid, scoring='f1')
clf = clf.fit(features_train,labels_train)

best_kneighbors = clf.best_estimator_
classifiers.append(best_kneighbors)

param_grid = {"criterion": ["gini", "entropy"],
              "min_samples_split": [2,3,5],
              "max_depth": [2,3,5,10],
              "min_samples_leaf": [2, 5, 10,15,20],
              "random_state": [None, 5],
              "presort": [True],
              'class_weight':['balanced',{True:9,False:1},{True:8,False:2},{True:7,False:3},{True:6,False:4}]
              }

clf = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring='f1')
clf = clf.fit(features_train,labels_train)

best_decision_tree = clf.best_estimator_
classifiers.append(best_decision_tree)

### FINAL CLASSIFIER SELECTED ###
import pdb; pdb.set_trace()

clf = classifiers[4]

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
