# Test file for processing.py

from __future__ import print_function

import numpy as np

import processing as proc

def test_convert_gender():
   """
   1
   0
   2
   """
   print('Female\t\t', proc.convert_gender('Female'))
   print('Male\t\t', proc.convert_gender('Male'))
   print('Unknown\t\t', proc.convert_gender('Unknown'))


def test_convert_language():
   """
   2
   1
   0
   """
   print('EnglishandAnother', proc.convert_language('EnglishandAnother'), sep='')
   print('English\t\t', proc.convert_language('English'))
   print('Another\t\t', proc.convert_language('Another'))
   
   
if __name__ == '__main__':

   test_convert_gender()
   test_convert_language()
