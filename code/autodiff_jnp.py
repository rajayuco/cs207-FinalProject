



###TRY: RECALL FOR EVERY KEY COMBINATION IN AUTODIFF? AS IN, RECORD THE UNIQUE COMBOS AND PASS ON?
import numpy as np


class autodiff():
	def __init__(self,name,val,der=1):
		self.name = name
		self.val = val
		self.der = {name:der}
	
	
	#FUNCTION: __add__
	#PURPOSE: Add this autodiff instance to a number or to another autodiff instance, and calculate the derivatives resulting from this action.
	def __add__(self, other):
		#Generate a new autodiff instance copy of self
		anew = autodiff(self.name, self.val, self.der)
		
		#Tries adding two autodiff instances together
		try:
			#Add values
			anew.val = self.val + other.val
			
			#Calculate derivatives of this addition for all variables so far encountered
			for key in np.unique([key for key in self.der] + [key for key in other.der]): #Iterate through all unique variables so far encountered
				#If self has not encountered this variable before (so derivative of self with respect to variable must be 0)
				if key not in self.der:
					anew.der[key] = other.der[key]
				
				#Else, if opponent has not encountered this variable before (so derivative of self with respect to variable must be 0)
				elif key not in other.der:
					anew.der[key] = self.der[key]
				
				#Else, if both self and opponent have encountered this variable before
				else:
					anew.der[key] = self.der[key] + other.der[key]
		
		#Otherwise, if not two autodiff instances:
		except AttributeError:
			#Tries adding autodiff instance and number together
			try:
				for key in self.der:
					anew.der[key] = self.der[key]
					anew.val = other + self.val
			
			#Otherwise, raises a type error if not compatible
			except:
				raise TypeError('please input a number or autodiff class')
		
		#Returns new autodiff instance
		return anew
	
	
	
	#FUNCTION: __radd__
	#PURPOSE: Allows commutative addition.
	def __radd__(self, other):
		return self + other
	
	
	
	#FUNCTION: __sub__
	#PURPOSE: Subtract an autodiff instance or number from a autodiff instance, and calculate the derivatives resulting from this action.
	def __sub__(self, other):
		#Generate a new autodiff instance copy of self
		anew = autodiff(self.name, self.val, self.der)
		
		#Tries subtracting two autodiff instances together
		try:
			#Subtract values
			anew.val = self.val - other.val
			
			#Calculate derivatives of this subtraction for all variables so far encountered
			for key in np.unique([key for key in self.der] + [key for key in other.der]): #Iterate through all unique variables so far encountered
				#If self has not encountered this variable before (so derivative of self with respect to variable must be 0)
				if key not in self.der:
					anew.der[key] = -1*other.der[key]
				
				#Else, if opponent has not encountered this variable before (so derivative of self with respect to variable must be 0)
				elif key not in other.der:
					anew.der[key] = self.der[key]
				
				#Else, if both self and opponent have encountered this variable before
				else:
					anew.der[key] = self.der[key] - other.der[key]
		
		#Otherwise, if not two autodiff instances:
		except AttributeError:
			#Tries subtracting number from autodiff instance
			try:
				for key in self.der:
					anew.der[key] = self.der[key]
					anew.val = self.val - other
			
			#Otherwise, raises a type error if not compatible
			except:
				raise TypeError('please input a number or autodiff class')
		
		#Returns new autodiff instance
		return anew
	
	
	
	#FUNCTION: __sub__
	#PURPOSE: Subtract an autodiff instance or number from a autodiff instance, and calculate the derivatives resulting from this action.
	def __rsub__(self, other):
		return (-1*self) + other
	
	
	
	#FUNCTION: __pow__
	#PURPOSE: Raise this autodiff instance to a number or to another autodiff instance, and calculate the derivatives resulting from this action.
	def __pow__(self, other):
		#Generate a new autodiff instance copy of self
		anew = autodiff(self.name, self.val, self.der)
		
		#Tries raising this autodiff instance to another autodiff instance
		try:
			#Raise values
			anew.val = self.val**other.val
			
			#Calculate derivatives of this exponentiation for all variables so far encountered
			for key in np.unique([key for key in self.der] + [key for key in other.der]): #Iterate through all unique variables so far encountered
				#If self has not encountered this variable before (so derivative of self with respect to variable must be 0)
				if key not in self.der:
					anew.der[key] = anew.val*(np.log(self.val)*other.der[key])
				
				#Else, if opponent has not encountered this variable before (so derivative of self with respect to variable must be 0)
				elif key not in other.der:
					anew.der[key] = anew.val*(other.val*self.der[key]/1.0/self.val)
				
				#Else, if both self and opponent have encountered this variable before
				else:
					anew.der[key] = anew.val*((np.log(self.val)*other.der[key]) + (other.val*self.der[key]/1.0/self.val))
		
		#Otherwise, if not two autodiff instances:
		except AttributeError:
			#Tries adding autodiff instance and number together
			try:
				for key in self.der:
					anew.der[key] = other*(self.val**(other - 1))*self.der[key]
					anew.val = self.val**other
			
			#Otherwise, raises a type error if not compatible
			except:
				raise TypeError('please input a number or autodiff class')
		
		#Returns new autodiff instance
		return anew
	
	
	
	def __rpow__(self, other):
		return self**other
	
	
	
	def __mul__(self,other):
		#assuming other is autodiff class
		anew = autodiff(self.name, self.val, self.der)
		try:
			anew.val = self.val*other.val
			for key in np.unique([key for key in self.der] + [key for key in other.der]):
				if key not in self.der:
					anew.der[key]=self.val*other.der[key]
				elif key not in other.der:
					anew.der[key]=self.der[key]*other.val
				else:
					anew.der[key]=self.der[key]*other.val+other.der[key]*self.val        
		except AttributeError:
			try:
				for key in self.der:
					anew.der[key] = other*self.der[key]
					anew.val = other*self.val
			except:
				raise TypeError('please input a number or autodiff class')
		return anew

	__rmul__ = __mul__