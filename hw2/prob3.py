
import sys, math


def NBgausHonor( gpa ):
      mean = 3.4
      vari = 0.63
      
      const = float(1) / math.pow( 2 * math.pi * vari, .5 )
      
      epower = math.exp(-1 * math.pow( gpa - mean, 2 ) / ( 2 * vari ))
      
      result = const * epower
      
      return result

def NBgausNormal( gpa ):
      mean = 3.083
      vari = 0.1737
      
      const = float(1) / math.pow( 2 * math.pi * vari, .5 )
      
      epower = math.exp(-1 * math.pow( gpa - mean, 2 ) / ( 2 * vari ))
      
      result = const * epower
      
      return result 

if __name__ == '__main__':

   #gpa = sys.argv[1]
   #ap = sys.argv[2]
   #nh = sys.argv[3]
   
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
   nho = float(5)/9
   
   for val in xrange(1,420):
      gpa = float(val)/100
      if (pho * NBgausHonor( gpa ) * hap) > (nho * NBgausNormal( gpa ) * nap ):
         print gpa, " GPA Honors**, AP Yes"
      else:
         print gpa, " GPA Normal, AP Yes"
         
   print "\n\n"
   for val in xrange(1,420):
      gpa = float(val)/100
      if (pho * NBgausHonor( gpa ) * hno) > (nho * NBgausNormal( gpa ) * nno ):
         print gpa, " GPA Honors**, AP No"
      else:
         print gpa, " GPA Normal, AP No"
      
   