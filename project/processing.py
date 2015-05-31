from __future__ import print_function

import sys
import math
import argparse
import numpy as np
from operator import itemgetter
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import Imputer
from sklearn.dummy import DummyClassifier


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
   
   # Fix the instances weirdness
   instance_list = []
   for idx,instance in enumerate(instances):
      instance_list.append( [ value for value in instance ] ) 
   bandaid = Imputer( strategy='median' )
   instances = bandaid.fit_transform( instance_list )
                           
   dataset = np.genfromtxt(filename, delimiter = ',', names = True,
                           converters = {'gender': lambda s: convert_gender(s), 
                           'FirstLang': lambda s: convert_language(s)}
                            )
                           
   return dataset, instances, outcomes
   

def generate_labels( outcomes ):
   """
   Use KMeans clustering to find a lowest performing group and label them as failures for life
   
   2015/5/31: nan values are being counted as pass
   """

   # Create array of just GPA data
   gpas = [ [gpa] for gpaunits, cumunits, gpa in outcomes ]
   
   # Replace nan with 13, so that all nan will be given their own cluster
   # Keep track of where the nan are, so that we can verify their cluster (TEST)
   for idx,gpa in enumerate( gpas ):
      if math.isnan( gpa[0] ):
         gpas[idx] = [np.float64( 13 )]

   # Fit the clusters
   cluster_count = 3
   kmeans = KMeans( n_clusters = cluster_count )
   kmeans_gpa = kmeans.fit( gpas )
   
   # Creates ( label, GPA ) list
   label_gpa = zip( kmeans_gpa.labels_, gpas ) 
   
   # Which number corresponds to the lowest GPAs?
   min_label = min( label_gpa, key=itemgetter(1) )[0]
   
   # Change numeric labels to min == "fail" and not min == "pass"
   for idx,(label,gpa) in enumerate( label_gpa ):
      if label == min_label:
         label_gpa[idx] = ( "fail", gpa )
      else:
         label_gpa[idx] = ( "pass", gpa )

   labels = [ label for label,gpa in label_gpa ]

   return labels

   
def NBclassify( instances, labels ):
   """
   Create a naive bayes classifier from the input data
   """
   clf = GaussianNB()
  
   classifier = clf.fit( instances, labels )
   
   return classifier
   
   
def evaluate( clf, dumb_clf, instances, labels ):
   """
   Evaluate the classifier
   """
   

   baseline = dumb_clf.score( instances, labels )
   print( "Baseline: ", baseline )
   
   print( "Naive Bayes: ", clf.score(instances, labels) )

   
if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--file", help="the name of the data file to process",
                       type=str, required=True)
   parser.add_argument("--action", help="'train' for training and 'test' for testing",
                       type=str, required=True)
   args = parser.parse_args()
   filename = args.file
   action = args.action

   
   # Get the full data set, instances, and outcomes.
   dataset, instances, outcomes = get_data( filename )

   # Generate labels array from the outcome data
   labels = generate_labels( outcomes )

   # Split data into training and dev sets
   size_of_test_set = 0.2
   instance_train, instance_test, labels_train, labels_test =\
      train_test_split( instances, labels, test_size = size_of_test_set )
   
   assert len(instance_train) == len(labels_train) and len(instance_test) == len(labels_test)
   
   # Classify the training set
   classifier = NBclassify( instance_train, labels_train )
   
   # Baseline
   baseline = DummyClassifier( strategy='uniform' )
   dumb_clf = baseline.fit( instance_train, labels_train )

   # Evaluate the classification
   evaluate( classifier, dumb_clf, instance_test, labels_test )

   
