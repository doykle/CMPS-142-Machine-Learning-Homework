"""

Things to do:
   Balance the training data
   
   Conditional Frequency
   
   Separating by ACT and SAT (or generally separating by data presence)

2015/6/3 Notes:;
   implementing PCA
   need to standardize the data??
   the doing logistic regression
   
   
"""



from __future__ import print_function

import sys
import math
import argparse
import numpy as np
from operator import itemgetter
import pdb
import threading

# Unsupervised Clusters
from sklearn.cluster import KMeans
# Separation of training into training/test set
from sklearn.cross_validation import train_test_split
# Gaussian Naive Bayes classification
from sklearn.naive_bayes import GaussianNB
# Fill in missing values. Handles NaN values by filling in values
from sklearn.preprocessing import Imputer
# Calculates baseline for classification
from sklearn.dummy import DummyClassifier
# Calculate/display evaluation metrics
from sklearn.metrics import confusion_matrix, classification_report
# Dimensionality Reduction
from sklearn import decomposition
# For using logistic regression
from sklearn.linear_model import LogisticRegression
# Grid search for optimizing parameters
from sklearn.grid_search import GridSearchCV
# SVM Regression, Classification
from sklearn.svm import SVR, SVC
# For pipeline
from sklearn.pipeline import Pipeline
# Scaling data
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import matplotlib.cm as cm


ALL_FEATURES = ["Subjnum", "gender", "Firgen", "famincome",	"SATCRDG",	"SATMATH",	"SATWRTG",	"SATTotal",	"HSGPA",	"ACTRead",	"ACTMath",	"ACTEngWrit",	"APIScore",	"FirstLang",	"HSGPAunweighted"] #	Firststyrunitsforgpa	Firststyeartotcumunits	Firstyrcumgpa]


def convert_gender( s ):
   """
   Turn string 'gender' attribute into int
   """
   if s == 'Female':
      return 1
   elif s == 'Male':
      return 0
   else:
      return 2

def convert_language( s ):
   """
   Turn string 'language' attribute into int
   """
   if s == 'EnglishandAnother':
      return 2
   elif s == 'English':
      return 1
   else:
      return 0
   

def get_data( filename ):
   """
   Extract the full dataset, once as an entire set, and again 
   as separate instances,labels
   """
   outcomes = np.genfromtxt(filename, delimiter = ',', names = True,
                           usecols=xrange(15,18) )
                           
   instances = np.genfromtxt(filename, delimiter = ',', names = True,
                           converters = {'gender': lambda s: convert_gender(s), 
                           'FirstLang': lambda s: convert_language(s)},
                           usecols=xrange(15) )
                           
   dataset = np.genfromtxt(filename, delimiter = ',', names = True,
                           converters = {'gender': lambda s: convert_gender(s), 
                           'FirstLang': lambda s: convert_language(s)}
                            )
                           
   return dataset, instances, outcomes
   
   
def get_test_data( filename ):
   """
   Extract the full dataset, once as an entire set, and again 
   as separate instances,labels
   """

   subject_nums = np.genfromtxt(filename, delimiter = ',', names = True,
                           usecols=0 )
                           
   
   instances = np.genfromtxt(filename, delimiter = ',', names = True,
                           converters = {'gender': lambda s: convert_gender(s), 
                           'FirstLang': lambda s: convert_language(s)},
                           usecols=xrange(15) )
                           
                           
   return subject_nums, instances
   

def data_organizer( instances, outcomes ):
   """
   Operations to organize data as desired
   """
   
   excluded_features = set([])
   #print( "Using only SAT subject tests" )
   #included_features = set(["SATCRDG",	"SATMATH",	"SATWRTG"])
   
   #print( "Using SAT total and HSGPA" )
   #included_features = set(["SATTotal",	"HSGPA"])
   
   #print( "Using gender, firstgen, famincome, firstlang" )
   #included_features = set(["gender", "Firgen", "famincome", "FirstLang"])
   
   print( "Using all features" )
   included_features = set(["gender", "Firgen", "famincome",	"SATCRDG",	"SATMATH",	"SATWRTG",	"SATTotal",	"HSGPA",	"ACTRead",	"ACTMath",	"ACTEngWrit",	"APIScore",	"FirstLang",	"HSGPAunweighted"])

   #print( "SAT subject tests and HSGPA" )
   #included_features = set(["SATCRDG",	"SATMATH",	"SATWRTG", "HSGPA" ])


   # Remove instances without GPA data
   new_instances = []
   new_outcomes = []
   for instance,outcome in zip(instances,outcomes):
      temp={}
      for name,val in zip(ALL_FEATURES, instance):
         temp[name] = val
      u1,u2,gpa = outcome
      if not math.isnan( gpa ):
         temp_list = []
         skip = False
         for key in temp.keys():
            if key in included_features:
               if math.isnan(temp[key]):
                  skip = True
               temp_list.append( temp[key] )
         if not skip:
            new_outcomes.append( [value for value in outcome] )
            new_instances.append( temp_list )
         
         
   instances = new_instances
   outcomes = new_outcomes

   
   # Fill in NaN values with median
   instance_list = []
   for idx,instance in enumerate(instances):
      instance_list.append( [ value for value in instance ] ) 
   bandaid = Imputer( strategy='median' )
   instances = bandaid.fit_transform( instance_list )
   
   # Scale to [0,1]
   scaler = MinMaxScaler( feature_range=(0,1), copy=False)
   instances = scaler.fit_transform(instances)

   return instances, outcomes
   
   
def data_organizer_two( instances, outcomes, feature ):
   """
   Operations to organize data as desired
   """
   
   excluded_features = set([])

   print( "Using all features" )
   included_features = set([feature])

   # Remove instances without GPA data
   new_instances = []
   new_outcomes = []
   for instance,outcome in zip(instances,outcomes):
      temp={}
      for name,val in zip(ALL_FEATURES, instance):
         temp[name] = val
      gpa = outcome
      if not math.isnan( gpa ):
         temp_list = []
         skip = False
         for key in temp.keys():
            if key in included_features:
               if math.isnan(temp[key]):
                  skip = True
               temp_list.append( temp[key] )
         if not skip:
            new_outcomes.append( [outcome] )
            new_instances.append( temp_list )
         
         
   instances = new_instances
   outcomes = new_outcomes

   return instances, outcomes
   
  
def visualize( km, n_clusters, X, y ):
 
   fig = plt.figure()
   plt.plot( X, y, 'ro')
   plt.show()

# http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
#label,gpa = list
def get_label_order( list ):
   list = sorted( list, key=itemgetter(1) )
   seen = set()
   seen_add = seen.add
   label_order = [ label for label,gpa in list if not (label in seen or seen_add(label)) ]
   label_dic = {}
   for idx,label in enumerate(label_order):
      label_dic[label] = idx+1
   return label_dic
   
   
def cluster_labels( values, clusters ):
   """
   Use KMeans clustering to find a lowest performing group and label them as failures for life
   
   """
   #pdb.set_trace()
   # Create array of just GPA data
   #val_array = []
   #for val in values:
   #   val_array.append([val])
     
   #values = val_array
   try:
      values = [ [value[0]] for value in values ]
   except IndexError:
      values = [ [value] for value in values ]
      
      
   # Fit the clusters
   cluster_count = clusters
   kmeans = KMeans( n_clusters = cluster_count )
   kmeans_gpa = kmeans.fit( values )
   
   # Creates ( label, GPA ) list
   label_val = zip( kmeans_gpa.labels_, values ) 
   
   label_dic = get_label_order( label_val )  

   for idx,(label,val) in enumerate( label_val ):
      label_val[idx] = ( label_dic[label], val )
   
   #visualize( kmeans, cluster_count, gpas, kmeans_gpa.labels_ )

   labels = [ label for label,val in label_val ]

   return labels
   

def split_at_two( outcomes ):
   """
   Use KMeans clustering to find a lowest performing group and label them as failures for life
   
   2015/5/31: nan GPA values are being counted as pass
   """

   # Create array of just GPA data
   gpas = [ [gpa] for gpaunits, cumunits, gpa in outcomes ]
   
   labels = []
   fail_count = 0
   pass_count = 0
   for gpa in gpas:
      if gpa[0] > 3.0:
         pass_count += 1
         labels.append(1)
      else:
         fail_count += 1
         labels.append(0)

   print( "below threshold: ", fail_count, " above: ", pass_count)

   return labels   

 
def generate_labels( outcomes ):
   """
   Use KMeans clustering to find a lowest performing group and label them as failures for life
   
   2015/5/31: nan GPA values are being counted as pass
   """

   # Create array of just GPA data
   gpas = [ [gpa] for gpaunits, cumunits, gpa in outcomes ]
   
   # Replace nan with 13, so that all nan will be given their own cluster
   # Keep track of where the nan are, so that we can verify their cluster (TEST)
   for idx,gpa in enumerate( gpas ):
      if math.isnan( gpa[0] ):
         gpas[idx] = [np.float64( 13 )]

   # Fit the clusters
   cluster_count = 2
   kmeans = KMeans( n_clusters = cluster_count )
   kmeans_gpa = kmeans.fit( gpas )
   
   # Creates ( label, GPA ) list
   label_gpa = zip( kmeans_gpa.labels_, gpas ) 
   
   label_dic = get_label_order( label_gpa )  

   for idx,(label,gpa) in enumerate( label_gpa ):
      label_gpa[idx] = ( label_dic[label], gpa )
   
   #visualize( kmeans, cluster_count, gpas, kmeans_gpa.labels_ )
   """
   # Which number corresponds to the lowest GPAs?
   min_label = min( label_gpa, key=itemgetter(1) )[0]
   
   # DBUG find threshold between pass/fail?
   
   # Change numeric labels to min == "fail" and not min == "pass"
   for idx,(label,gpa) in enumerate( label_gpa ):
      if label == min_label:
         label_gpa[idx] = ( "fail", gpa )
      else:
         label_gpa[idx] = ( "pass", gpa )
   """
   labels = [ label for label,gpa in label_gpa ]

   return labels

   
def NBclassify( instances, labels ):
   """
   Create a naive bayes classifier from the input data
   """
   clf = GaussianNB()
  
   classifier = clf.fit( instances, labels )
   
   return classifier
   

def logistic_regression( instances, labels ):
   """
   Logistic regression magic happens here
   
   building off the example found at:
   http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#example-plot-digits-pipe-py
   """
   logistic = LogisticRegression()
   
     
def svm_regression( instances, labels ):
   """
   SVM regression
   
   http://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html
   """
   #pdb.set_trace()
   X = np.array(instances)
   y = np.array(labels)
   
   #X= [ [x] for x in cluster_labels(X, 80) ]
   #y=cluster_labels(y, 10)
         
   print( X[:5] )
   print( y[:5] )
         
   #pdb.set_trace()
   ###############################################################################
   # Fit regression models
   svr_rbf = SVR(kernel='rbf', C=10, gamma=0.5)
   svr_lin = SVR(kernel='linear', C=.1)
   #svr_poly = SVR(kernel='poly', C=.1, degree=2)
   
   print( "RBF kernel" )
   y_rbf = svr_rbf.fit(X, y).predict(X)
   #print( "Linear kernel" )
   y_lin_fit = svr_lin.fit(X, y)
   y_lin = y_lin_fit.predict(X)
   #print( "Polynomial kernel" )
   #y_poly = svr_poly.fit(X, y).predict(X)
   print( "finished SVM training" )

   print( "Linear Kernel coefficients:\n" )
   #pdb.set_trace()
   for thing in y_lin_fit.coef_:
      for t,l in zip( thing, ALL_FEATURES ):
         print( l,t )
 
   ###############################################################################
   # look at the results
   x_axis = [ x*.1 for x in xrange(-1,11) ] 
   
   plt.scatter(x_axis, x_axis, c='k', label='data')
   plt.hold('on')
   plt.plot(X, y_rbf, label='RBF model')
   #plt.plot(X, y_lin, c='r', label='Linear model')
   #plt.plot(X, y_poly, c='b', label='Polynomial model')
   plt.xlabel('data')
   plt.ylabel('target')
   plt.title('Support Vector Regression')
   plt.legend()
   plt.show()


def svm_classification( instances, labels ):
   """
   SVM Classification
   
   for PCA pipeline
   http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#example-plot-digits-pipe-py
   """
   X = np.array(instances)
   y = np.array(labels)

   print( "training size: ", len(instances) )
   
   #param_grid = [
   #   {'C': [.1, 1, 5, 10, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 1000], 
   #    'gamma': [0.0001, 0.005, 0.001, 0.05, 0.01, 0.02, 0.04, 0.05, 0.07, 0.09, 0.095, 0.1, 0.13, 0.16, 0.19, 0.3, 0.6, 1], 'kernel': ['rbf'] }
   #  ]
   clf = SVC(kernel='linear')

   #pca = decomposition.PCA()
   #pipe = Pipeline(steps=[('pca',pca),('svm',svm)])
   
   #pca.fit(X)
   
   #pca_components = np.logspace(1, 3, 2).tolist()
   #pca_components = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
   
   C_vals = np.logspace(0.01, 1.5, 50).tolist()
   #g_vals = np.logspace(-5, 5, 20).tolist()
   #param_dict = dict( C=C_vals )
   
   
   
   #clf = GridSearchCV(svm, param_dict, verbose=True)
   
   #print( clf )

   #clf = SVC(kernel='rbf', C=22.252, gamma=0.007848)
   #clf = SVC(kernel='linear', C=489185, gamma=0.001281)
   #clf = SVC(kernel='rbf', C=10.75422, gamma=6.15848)
  
   clf.fit(X, y)
     
   #print( "best estimator: ", clf.best_estimator_ )
   
   #print( "grid scores: ", clf.grid_scores_ )
     
   #print("The best parameters are %s with a score of %0.2f"
   #   % (clf.best_params_, clf.best_score_))
     
   return clf
   
   
def knn_classification( instances, labels ):

   knn = KNeighborsClassifier(n_neighbors=9)
   clf = knn.fit(instances, labels)
   
   return clf
   
     
def evaluate( clf, dumb_clf, instances, labels ):
   """
   Evaluate the classifier
   """
   #target_names = ['class 1', 'class 2', 'class 3', 'class 4', 'class 5', 'class 6']
   #pdb.set_trace()
   #baseline = dumb_clf.score( instances, labels )
   #print( "Baseline: ", baseline )
   #prediction_array = dumb_clf.predict( instances )
   #conf_mat = confusion_matrix( labels, prediction_array )
   #class_report = classification_report( labels, prediction_array,\
   #                  target_names=target_names)
                     
   #print( "Classification report:\n", class_report )
   
   #print( "Confusion Matrix:\n", conf_mat )   
   baseline = .5
   #print( "\n\n" )
   
   class_score = clf.score(instances, labels)
   
   #print( "Trained Classifier: ", clf.score(instances, labels) )
   #prediction_array = clf.predict( instances )
   #conf_mat = confusion_matrix( labels, prediction_array )
   #class_report = classification_report( labels, prediction_array,\
   #                  target_names=target_names)
                     
   #print( "Classification report:\n", class_report )
   
   #print( "Confusion Matrix:\n", conf_mat )
   
   return (class_score, baseline)
   
   
def test_classifier( clf, instances ):
   labels = np.array(clf.predict(instances))
   #print( labels )
   count_1 = 0
   count_0 = 0
   for val in labels:
      if val:
         count_1 += 1
      else:
         count_0 += 1
   
   print(count_1, count_0)
   
   return labels

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--file", help="the name of the data file to process",
                       type=str, required=True)
   parser.add_argument("--action", help="'train' for training and 'test' for testing",
                       type=str, required=True)
   args = parser.parse_args()
   filename = args.file
   action = args.action

   if action == 'train':
      # Get the full data set, instances, and outcomes.
      dataset, instances, outcomes = get_data( filename )
      
      #pdb.set_trace()
      
      # Organize the data
      instances, outcomes = data_organizer( instances, outcomes )

      assert len(instances) == len(outcomes)
      
      # Generate labels array from the outcome data
      #labels = generate_labels( outcomes )
      labels = split_at_two( outcomes )
      size_of_test_set = 0.25
      instance_train_set, instance_test_set, labels_train_set, labels_test_set =\
            train_test_split( instances, labels, test_size = size_of_test_set, random_state=5 )
      
      output_strings = []
      output_strings.append("train_data_percent,test_data_percent,train_instance_count,,clf_train_acc,clf_test_acc,base_train_acc,base_test_acc" )
      # Split data into training and dev sets
      for feature in ALL_FEATURES:
         instance_train_set, labels_train_set = data_organizer_two( instance_train_set, labels_train_set, feature )
         instance_test_set, labels_test_set = data_organizer_two( instance_test_set, labels_test_set, feature )
         for num in xrange(40, 990, 10):
            size_of_test_set = 1-num*0.001
            #print( "Percent split: train--{0} test--{1}".format( 1-size_of_test_set, size_of_test_set ) )
            instance_train, instance_test, labels_train, labels_test =\
               train_test_split( instance_train_set, labels_train_set, test_size = size_of_test_set, random_state=5 )
            
            #print( len(instance_train), len(labels_train), len(instance_test), len(labels_test) )
            assert len(instance_train) == len(labels_train) and len(instance_test) == len(labels_test)

            # Classify the training set
            #classifier = NBclassify( instance_train, labels_train )
            #classifier = knn_classification( instance_train, labels_train )
            #classifier = logistic_regression( instance_train, labels_train )
            #classifier = svm_regression( instance_train, labels_train )
            classifier = svm_classification( instance_train, labels_train )
            
            
            
            # Baseline
            baseline = DummyClassifier( strategy='uniform' )
            dumb_clf = baseline.fit( instance_train, labels_train )

            # Evaluate the classification
            #print( "TRAINING EVAL" )
            clf_train_eval, base_train_eval = evaluate( classifier, dumb_clf, instance_train, labels_train)
            #print( "TEST EVAL" )
            clf_test_eval, base_test_eval = evaluate( classifier, dumb_clf, instance_test_set, labels_test_set )
            #print( "\n\n" )
            
            output_strings.append( "{0},{1},{2},{3},{4},{5},{6},{7}".format(1-size_of_test_set, size_of_test_set, len(instance_train),'', clf_train_eval, clf_test_eval, base_train_eval, base_test_eval) )
            
            
         for string in output_strings:
            print( string )
      # Get the full data set, instances, and outcomes.
      #subject_nums, instances = get_test_data( 'goodtestNoLabels.csv' )

      # Organize the data
      #instances = test_data_organizer( instances )
      
      # Get the predictions list
      #predictions = test_classifier( classifier, instances )
      
      #for num,pred in zip(subject_nums, predictions):
      #   print( int(num[0]), pred, sep=',')
         
      #print( len(subject_nums), len(predictions) )
      