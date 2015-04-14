# 
# Written by Kevin Doyle (kdoyle@ucsc.edu)
# 
# CMPS 142
# Homework 1, problem 5(a)
#
# This is an implementation of Stochastic Gradient descent
# using a boolean matrix designed according to the homework.
####

import random, operator

# Build an nxm boolean matrix with balanced entries
class HyperCube( object ):
   def __init__( self, instances, features, labels = True ):
      self.matrix = []
      self.labels = []
      self.build_matrix( instances, features )
      if labels:
         self.build_labels()
   
   def build_matrix( self, instances = 500, features = 11 ):
      rows = 0
      columns = 0
      
      number_pool = make_pool( instances * features )
      random.shuffle( number_pool )
      
      while rows < instances:
         columns = 0
         row = []
         while columns < features:
            row.append( number_pool.pop( get_number( len( number_pool ) ) ) )
            columns = columns + 1
         self.matrix.append( row )
         rows = rows + 1
         
   def build_labels( self ):
      for row in self.matrix:
         if row[0] == 1:
            self.labels.append( row[0] )
         else:
            self.labels.append( -1 )
           
# To achieve balance, we will select entry values from a 
# carefully populated, balanced array of boolean values.
def make_pool( size ):
   q = size / 2
   r = size % 2
   numbers = []
   
   counter = 0
   while counter < q:
      counter = counter + 1
      numbers.append( -1 )
   counter = 0
   while counter < q+r:
      counter = counter + 1
      numbers.append( 1 )
   
   return numbers
 
# The random number generator is used to select indices from 
# the boolean value array.
def get_number( range ):
   val = int( random.random()*1000 % range )
   return val

# Checking if the gradient descent has converged by monitoring
# rate of change in the calculated error values.
def yh_check( new, vals ):
   steady = True
   vals.pop(0)
   vals.append(new)
   l = len(vals)-1
   avg = sum([ val/l for val in vals ])
   for val in vals:
      if abs(avg-val) > 0.001:
         steady = False
   return (steady, vals)

# The main implementation of Stochastic Gradient descent. 
# This is where we use our training data to build the 
# theta vector. 
def train( step_size ):
   tr = training.matrix
   trl = training.labels
   theta = [0,0,0,0,0,0,0,0,0,0,0]
   yh_values = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
   counter = 0
   
   # Using one row at a time to adjust the theta vector
   for row,label in zip(tr,trl):
      counter = counter + 1
      
      # Calculating [01X1 + 02X2 + ... + 0nXn]
      h = sum( [( th * x ) for th,x in zip(theta,row) ] )
      
      # The error calculation: y - h(x)
      yh = label - h
      
      # Checking to see if the error is consistently minimal
      b, yh_values = yh_check( yh, yh_values )
      if b:
         break
      
      #print h, '\t', yh, '\t', theta
      
      # Adjusting each value of theta in the vector
      for idx,(th,x) in enumerate( zip(theta,row) ):
         theta[idx] = th + step_size * ( ( yh ) * x )
 
   return counter, theta
   
# Determine the accuracy of the built theta vector
def evaluate( theta ):
   predictions = []
   for row in test.matrix:
      predictions.append( sum([ th * x for th,x in zip(theta,row)]) )
  
   # Comparing the calculated prediction values against
   # known labels from the test data.
   c = 0
   x = 0
   for y,p in zip(test.labels,predictions):

      if (y == 1 and p > 0) or (y == -1 and p <= 0):
         #print y, p, "YES"
         c = c + 1
      else:
         #print y, p, "NOOOO"
         x = x + 1

   #print "Correct: ", c, " Incorrect: ", x
   return c
   
if __name__ == '__main__':
   # Seeding the random function with a constant value
   # helps with debugging and allows for observing the 
   # affect of changing code
   #random.seed(0)
   
   # Build the test and training data sets
   training = HyperCube( 1000, 11 )
   test = HyperCube( 500, 11 )
   
   # In order to determine an ideal step size, we will 
   # train and test at every step size from 0.001 to 1.
   results = []
   for x in xrange(1,1000):
      step = float(x)/1000
      a,t = train( step )
      c = evaluate( t )
      results.append( (step, c, a, t) )
   
   # The results can be sorted for visual assessment.
   winners = sorted(results, key= operator.itemgetter(1,2), reverse=True)
   
   # Here we pick out the result which combines highest
   # accuracy and lowest number of rows used to train.
   max_c = 0
   min_a = 9999
   ideal_t = ''
   ideal_step = 99
   for step,c,a,t in winners:
      if c >= max_c and a < min_a:
         max_c = c
         min_a = a
         ideal_t = t
         ideal_step = step
         
   print max_c, min_a, ideal_step, ideal_t
  
   #for idx,row in enumerate(training.matrix):
   #   print "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}".format( training.labels[idx],row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10])
   #      print(value),
   #   print ''
