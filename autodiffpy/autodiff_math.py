# import packages
import numpy as np

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
		
	
def exp(ad):
    """Returns autodiff instance of sin(x)

    INPUTS
    =======
    ad: autodiff instance

    RETURNS
    ========
    anew: autodiff instance
       returns a new autodiff instance with updated value and derivative(s)

    EXAMPLES
    =========
    >>> x = autodiff('x', 10)
    >>> f1 = sin(x)
    >>> print(f1.val, fl.der)
    -0.5440211108893699 {'x': -0.8390715290764524}
    """
    try:
        anew = autodiff(name=ad.name, val = np.exp(self.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = self.der[key]*anew.val
			
        return anew
    except TypeError:
        print("Error: input should be autodiff instance only.")
		


def sin(ad):
    """Returns autodiff instance of sin(x)

    INPUTS
    =======
    ad: autodiff instance

    RETURNS
    ========
    anew: autodiff instance
       returns a new autodiff instance with updated value and derivative(s)

    EXAMPLES
    =========
    >>> x = autodiff('x', 10)
    >>> f1 = sin(x)
    >>> print(f1.val, fl.der)
    -0.5440211108893699 {'x': -0.8390715290764524}
    """
    try:
        anew = autodiff(name=ad.name, val = np.sin(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = np.cos(ad.val)*ad.der[key]
        return anew
    except TypeError:
        print("Error: input should be autodiff instance only.")

def cos(ad):
    """Returns autodiff instance of cos(x)

    INPUTS
    =======
    ad: autodiff instance

    RETURNS
    ========
    anew: autodiff instance
       returns a new autodiff instance with updated value and derivative(s)

    EXAMPLES
    =========
    >>> x = autodiff('x', 10)
    >>> f1 = cos(x)
    >>> print(f1.val, fl.der)
    -0.8390715290764524 {'x': 0.5440211108893699}
    """
    try:
        anew = autodiff(name=ad.name, val = np.cos(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = -np.sin(ad.val)*ad.der[key]
        return anew
    except TypeError:
        print("Error: input should be autodiff instance only.")

def tan(ad):
    """Returns autodiff instance of tan(x)

    INPUTS
    =======
    ad: autodiff instance

    RETURNS
    ========
    anew: autodiff instance
       returns a new autodiff instance with updated value and derivative(s)

    EXAMPLES
    =========
    >>> x = autodiff('x', 10)
    >>> f1 = tan(x)
    >>> print(f1.val, fl.der)
    0.6483608274590867 {'x': 1.4203717625834316}
    """
    try:
        anew = autodiff(name=ad.name, val = np.tan(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = 1/(np.cos(ad.val))**2*ad.der[key]
        return anew
    except TypeError:
        print("Error: input should be autodiff instance only.")

def log(ad):
        
        '''Returns autodiff instance of log(x)

        INPUTS
        ==========
        ad: autodiff instance

        RETURNS
        ==========
        anew: autodiff instance with updated values and derivatives

        EXAMPLES
        ==========
        >>> x = autodiff('x', np.exp(2))
        >>> f1 = log(x)
        >>> f1.val = 2.0
        >>> f1.der = 0.1353352832366127
        '''

        try:
            if ad.val<=0:
                raise ValueError
            anew = autodiff(name = ad.name, val = np.log(ad.val), der = ad.der)
            for key in ad.der:
                anew.der[key] = ad.der[key]/ad.val
            return anew
        except TypeError:
            print("Error: input should be autodiff instance")
        except ValueError:
            print('Error: cannot evaluate the log of a nonpositive number')
