# 
# Written by Alejandro Aguilar (aaguil10@ucsc.edu)
# 
# CMPS 142
# Homework 1, problem 5
#
# This is an implementation of Stochastic Gradient descent
#------------------------------------------------------------------------------- 
import numpy as np
from random import *

randBinList = lambda n: [np.float64(randint(0,1)) for b in range(1,n+1)]

def RandBinList(n):
	a = [0] * n
	for b in range(0,n-1):
		seed(None)
		a[b] = np.float64(randint(0,1))
	return a
	
def uniformBinList(n):
	a = [0] * n
	for b in range(0,n):
		seed(None)
		num = np.float64(uniform(0,1))
		if num > 0.5:
			a[b] = 1
		else:
			a[b] = -1
	return a
	
def update_theta(theta, trainingEx, learn_rate, costfunction):
	for i in range(0,11):
		theta[i] = theta[i] + ((learn_rate*costfunction(theta, trainingEx))*trainingEx[i])

#-------------------------------------------------------------------------------
		
def costfunctionA(theta, trainingEx):
	vect = theta*trainingEx
	h = 0
	for i in range(0,11):
		h += vect[i]
	cost = np.subtract(trainingEx[0],h)
	return cost
	
def test_thetaA(theta, testEx):
	threshold = .5
	vect = theta*testEx
	h = 0
	for i in range(0,11):
		h += vect[i]
	if testEx[0] - threshold < h < testEx[0] + threshold:
		return True
	else:
		return False

#-------------------------------------------------------------------------------		
def costfunctionB(theta, trainingEx):
	y = -64
	count1 = 0
	for i in range(0,11):
		if trainingEx[i] == 1:
			count1 = count1 + 1
	if count1 > 5:
		y = 1
	else:
		y = 0
	vect = theta*trainingEx
	h = 0
	for i in range(0,11):
		h += vect[i]
	cost = np.subtract(y,h)
	return cost
		
def test_thetaB(theta, testEx):
	y = -64
	count1 = 0
	for i in range(0,11):
		if testEx[i] == 1:
			count1 = count1 + 1
	if count1 > 5:
		y = 1
	else:
		y = 0
	threshold = .5
	vect = theta*testEx
	h = 0
	for i in range(0,11):
		h += vect[i]
	if y - threshold < h < y + threshold:
		return True
	else:
		return False

#-------------------------------------------------------------------------------

def costfunctionC(theta, trainingEx):
	r = int(uniform(-4,4))
	sum = 0
	y = -64
	for i in range(0,11):
		sum += trainingEx[i]
	if (r+sum) > 0:
		y = 1
	else:
		y = 0
	vect = theta*trainingEx
	h = 0
	for i in range(0,11):
		h += vect[i]
	cost = np.subtract(y,h)
	return cost
		
def test_thetaC(theta, testEx):
	r = int(uniform(-4,4))
	sum = 0
	y = -64
	for i in range(0,11):
		sum += testEx[i]
	if (r+sum) > 0:
		y = 1
	else:
		y = 0
	threshold = .25
	vect = theta*testEx
	h = 0
	for i in range(0,11):
		h += vect[i]
	if y - threshold < h < y + threshold:
		return True
	else:
		return False
		
def calc_avg(w):
	sum = [0,0,0,0,0,0,0,0,0,0,0]
	avg = [0,0,0,0,0,0,0,0,0,0,0]
	for i in range(0,1000):
		for j in range(0,11):
			sum[j] += w[i][j]
	for i in range(0,11):
		avg[i] = sum[i]/1000	
	return avg
	
def calc_avg2(w):
	sum = [0,0,0,0,0,0,0,0,0,0,0]
	avg = [0,0,0,0,0,0,0,0,0,0,0]
	for i in range(500,1000):
		for j in range(0,11):
			sum[j] += w[i][j]
	for i in range(0,11):
		avg[i] = sum[i]/500	
	return avg
	
#-------------------------------------------------------------------------------
	
training_set = []
test_set = []
theta = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]

#build training set
for i in range(0,500):
	g = uniformBinList(11)
	training_set.append(np.array(g))

#build test set		
for i in range(0,500):
	g = uniformBinList(11)
	test_set.append(np.array(g))	
	
#-----------------------------------Part A--------------------------------------

prediction_errors = 0
epochs = 1

#Training theta
for k in range(0,epochs):
	for i in range(0,500):
		update_theta(theta,training_set[i],.01,costfunctionA)
	
#test theta
for i in range(0,500):
	if test_thetaA(theta,test_set[i]) == False:
		prediction_errors = prediction_errors + 1

print ""
print "PART A"
print "Number of epochs: " + str(epochs)
print "prediction_errors: " + str(prediction_errors)

#-----------------------------------Part B--------------------------------------

prediction_errors = 0
epochs = 2
#Training theta
for k in range(0,epochs):
	for i in range(0,500):
		update_theta(theta,training_set[i],.01,costfunctionB)
	
#test theta
for i in range(0,500):
	if test_thetaB(theta,test_set[i]) == False:
		prediction_errors = prediction_errors + 1

print ""
print "PART B"
print "Number of epochs: " + str(epochs)
print "prediction_errors: " + str(prediction_errors)


#-----------------------------------Part C--------------------------------------

prediction_errors = 0
epochs = 2
w = []

#Training theta
for k in range(0,epochs):
	for i in range(0,500):
		update_theta(theta,np.array(training_set[i]),.0001,costfunctionC)
		w.append( np.array(theta) )
	
print ""
print "PART C"
#print "w_1000: " + str(w[999])
#print "avg w: " + str(calc_avg(w))
#print "avg w of second epoch: " + str(calc_avg2(w))

#log-likelihood = w*x see forum.
print "log-likelihood: " 
print str(w[999]*calc_avg(w))
print "log-likelihood of second epoch: "
print str(w[999]*calc_avg2(w))

