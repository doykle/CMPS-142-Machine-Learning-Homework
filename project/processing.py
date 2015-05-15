from __future__ import print_function

import argparse
import numpy as np


def convert_gender( s ):
   if s == 'Female':
      return 1
   elif s == 'Male':
      return 0
   else:
      return 2
      
  
def convert_language( s ):
   if s == 'EnglishandAnother':
      return 2
   elif s == 'English':
      return 1
   else:
      return 0
   
   
def get_dataset( filename ):

   dataset = np.genfromtxt(filename, delimiter = ',', names = True,
                           converters = {'gender': lambda s: convert_gender(s), 
                           'FirstLang': lambda s: convert_language(s)}
                           )
                           
   return dataset
   
   
def break_off_labels( dataset ): 
   
   labels = np.zeros(len(dataset['gender']), dtype=[('gpaunits', '<f8'),
                                                    ('units', '<f8'),
                                                    ('gpa', '<f8')
                                                    ])
   
   labels['gpa']      = dataset['Firstyrcumgpa'].copy()
   labels['units']    = dataset['Firststyeartotcumunits'].copy()
   labels['gpaunits'] = dataset['Firststyrunitsforgpa'].copy()
   
   #instances = np.delete(dataset, 'Firstyrcumgpa', 1)
   
      
      
   
   
if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--file", help="the name of the data file to process",
                       type=str, required=True)
   parser.add_argument("--action", help="'train' for training and 'test' for testing",
                       type=str, required=True)
   args = parser.parse_args()
   filename = args.file
   action = args.action

   
   # Get the data set
   # dataset is an indexable dictionary 
   dataset = get_dataset( filename )
   
   break_off_labels( dataset )
                     
   print(dataset.dtype)
   