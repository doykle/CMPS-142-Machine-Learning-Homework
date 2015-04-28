
import sys, math


def NBgausHonor( gpa ):
      mean = 3.4
      vari = 0.42
      
      const = float(1) / math.pow( 2 * math.pi * vari, .5 )
      
      epower = math.exp(-1 * math.pow( gpa - mean, 2 ) / ( 2 * vari ))
      
      result = const * epower
      
      print const, epower
      
      return result

def NBgausNormal( gpa ):
      mean = 3.0
      vari = 0.243333
      #0.292
      
      
      const = float(1) / math.pow( 2 * math.pi * vari, .5 )

      epower = math.exp(-1 * math.pow( gpa - mean, 2 ) / ( 2 * vari ))
      
      print const, epower
      
      result = const * epower
      
      return result 

if __name__ == '__main__':

   # Honors, AP
   hap = float(2)/3
   # Honors, no AP
   hno = float(1)/3
   
   # Normal, AP
   nap = float(2)/6
   # Normal, no AP
   nno = float(4)/6
   
   # Honors
   pho = float(3)/9
   # Normal
   nho = float(6)/9
   
   # Honors with AP
   # Math result: [1.67829, 3.21982]
   for val in xrange(1,420):
      gpa = float(val)/100
      if (pho * NBgausHonor( gpa ) * hap) > (nho * NBgausNormal( gpa ) * nap ):
         print gpa, ": Yes AP, Honors"
      else:
         print gpa, ": Yes AP, Normal"
         
   print "\n\n"
   
   # Honors without AP
   #Math result: [1.75831, 3.1398]
   for val in xrange(1,420):
      gpa = float(val)/100
      if (pho * NBgausHonor( gpa ) * hno) > (nho * NBgausNormal( gpa ) * nno ):
         print gpa, " No  AP, Honors"
      else:
         print gpa, " No  AP, Normal"
         
   