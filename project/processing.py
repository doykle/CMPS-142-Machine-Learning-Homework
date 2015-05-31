from __future__ import print_function

import argparse
import numpy as np


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
   

def generate_labels( outcomes ):
   """
   Use KMeans clustering to find a lowest performing group and label them as failures for life
   """
   

   
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

   # Classify the training set

   # Evaluate the classification
   

   
