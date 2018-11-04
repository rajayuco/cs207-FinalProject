# import packages
import numpy as np


class autodiff():
    def __init__(self,name,val,der=1):
        self.name = name
        self.val = val
        self.der = {name:der}

    def __neg__(self):
        """Allows unary operation of autodiff instance."""
        anew = autodiff(self.name, -self.val, self.der)
        for key in self.der:
            anew.der[key] = -self.der[key]
        return anew

    def __mul__(self, other):
        """Allows multiplication of another autodiff instance, or multiplication of a constant (integer or float)."""

        anew = autodiff(self.name, self.val, self.der)
        #assuming that other is autodiff instance
        try:
            anew.val = self.val*other.val
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key]=self.val*other.der[key]
                elif key not in other.der:
                    anew.der[key]=self.der[key]*other.val
                else:
                    anew.der[key]=self.der[key]*other.val+other.der[key]*self.val
        # if 'other' is not autodiff instance
        except AttributeError:
            # assuming that 'other' is a valid constant
            try:
                for key in self.der:
                    anew.der[key] = other*self.der[key]
                    anew.val = other*self.val
            except:
                raise TypeError('Error: please input a number or autodiff class.')
        return anew

    __rmul__ = __mul__

    def __truediv__(self,other):
        '''
        funtion for left division
        
        '''   
        anew = autodiff(self.name, self.val, self.der)
        try:
            anew.val = self.val/other.val
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key]=self.val*other.der[key]/other.val**2
                elif key not in other.der:
                    anew.der[key]=self.der[key]/other.val
                else:
                    anew.der[key]=self.der[key]/other.der[key]       
        except AttributeError:
            try:
                for key in self.der:
                    anew.der[key] = self.der[key]/other
                    anew.val = self.val/other
            except:
                raise TypeError('please input a number or autodiff class')
        return anew
  
    def __rdiv__(self, other):
        '''
        function for right division
        '''   
        anew = autodiff(self.name, self.val, self.der)
        try:
            anew.val = self.val/other.val
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key]=other.der[key]/self.val
                elif key not in other.der:
                    anew.der[key]=self.der[key]*other.val[key]/self.val**2
                else:
                    anew.der[key]=self.der[key]/other.der[key]       
        except AttributeError:
            try:
                for key in self.der:
                    anew.der[key] = self.der[key]/other
                    anew.val = self.val/2
            except:
                raise TypeError('please input a number or autodiff class')
        return anew
    
    
   