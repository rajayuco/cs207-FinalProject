import numpy as np
import pandas as pd
#from autodiff_math import *
from autodiffpy.autodiff_math import *

class autodiff():
    def __init__(self,name,val,der=1):
        self.name = name
        # set val attribute
        if isinstance(val, np.ndarray):
            self.val = val
        elif isinstance(val, list):
            self.val = np.asarray(val)
        else:
            self.val = np.asarray([val])

        if isinstance(der, np.ndarray):
            self.der = {name:der}
        elif isinstance(der, list):
            self.der = {name:np.asarray(der)}
        else:
            self.der = {name:np.asarray([der]*self.val.shape[0])}


        self.lparent = None
        self.rparent = None

        self.forwardpropcomplete = 'No'

        self.function = None

        self.back_der = None
        self.back_partial_der = None


    def __str__(self):
       return f"value: {self.val}\nderivatives:{self.der}"


    def __eq__(self, other):
           if isinstance(other, autodiff) == False:
               raise ValueError("Error: only autodiff instances can be compared with another.")
           if (False in (self.val == other.val)):
               return False
           if (False in (np.unique([key for key in self.der]) == np.unique([key for key in other.der]))):
               return False
           for key in self.der:
               if (False in (self.der[key] == other.der[key])):
                   return False
           return True


    def __ne__(self, other):
        return not (self == other)

    def __neg__(self,other=-1):
        """Allows unary operation of autodiff instance."""
        anew = autodiff(self.name, -self.val, self.der)
        anew.lparent = self
        for key in self.der:
            anew.der[key] = -1*self.der[key]
        self.back_partial_der = -1
        anew.function = self.__neg__
        return anew


    def __mul__(self, other):
        """Allows multiplication of another autodiff instance, or multiplication of a constant (integer or float)."""
        if isinstance(other, pd.DataFrame):
            other = np.asarray(other)
        if isinstance(other, (int, float, autodiff, list, np.ndarray)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be multiplied.")

        if isinstance(other, list):
            other = np.asarray(other)


        anew = autodiff(self.name, self.val, self.der)

        #Stores for backpropagation functionality
        anew.lparent = self
        anew.rparent = other

        anew.function=self.__mul__

        #for data/gradient descent
        #if

        #assuming that other is autodiff instance
        try:
            anew.val = self.val*other.val
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key]=self.val*(other.der[key])
                elif key not in other.der:
                    anew.der[key]=(self.der[key])*other.val
                else:
                    anew.der[key]=(self.der[key])*other.val+(other.der[key])*self.val

            #set the back partial derivatives that can be used for backpropagation
            self.back_partial_der = other.val
            other.back_partial_der = self.val


        # if 'other' is not autodiff instance
        except AttributeError:
            # assuming that 'other' is a valid constant
            if isinstance(other, (int,float)):
                anew.val = self.val*other
                for key in self.der:
                    anew.der[key] = other*self.der[key]
                self.back_partial_der = other

            else:
                other = np.asarray(other)
                if other.shape!=anew.val.shape:
                    anew.val = np.dot(other, self.val)
                else:
                    anew.val = self.val*other

                for key in self.der:
                    anew.der[key] = other
                    #anew.der[key] = np.dot(other,self.der[key])
                try:
                    fder = [[] for i in range(len(other[0]))]
                    for idx,value in enumerate(other):
                        for idx2,value2 in enumerate(value):
                            fder[idx2].append(value2)
                except:

                    fder = other

                self.back_partial_der = fder

        return anew

    __rmul__ = __mul__

    def __truediv__(self,other):
        '''function for left division'''

        if isinstance(other, (int, float, list, np.ndarray, autodiff)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be divided.")
        if isinstance(other,list):
            other = np.asarray(other)
        anew = autodiff(self.name, self.val, self.der)
        anew.lparent = self
        anew.rparent = other

        anew.function = self.__truediv__
        self.back_partial_der = 1/other
        try:
            anew.val = self.val/other.val
            self.back_partial_der = 1/other


            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key]= -self.val*other.der[key]/(other.val**2)
                elif key not in other.der:
                    anew.der[key]=self.der[key]/other.val
                else:
                    anew.der[key]=0

            self.back_partial_der = 1/other.val
            other.back_partial_der = -self.val/(other.val**2)

        except AttributeError:
            anew.val = self.val/other

            for key in self.der:
                anew.der[key] = (self.der[key])/other

                self.back_partial_der = 1/other

        return anew


    def __rtruediv__(self, other):
        '''function for right division'''

        if isinstance(other, (int, float, list, np.ndarray, autodiff)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be divided.")
        if isinstance(other,(list,float,int)):
            other = np.asarray(other)

        anew = autodiff(self.name, self.val, self.der)
        anew.lparent = self
        anew.rparent = other


        anew.function=self.__rtruediv__
        if isinstance(other, (int,float,list,np.ndarray)):

            for key in self.der:
                anew.der[key] = -other*(self.der[key])/self.val**2

            anew.val = other/self.val
            self.back_partial_der = -1*(self.val**2)

            return anew



    def __add__(self, other):

        if isinstance(other, (int, float, autodiff, list, np.ndarray)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be added.")


        #Generate a new autodiff instance copy of self
        anew = autodiff(self.name, self.val, self.der)

        anew.function=self.__add__

        anew.lparent = self
        anew.rparent = other


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

            self.back_partial_der = 1
            other.back_partial_der = 1

        #Otherwise, if not two autodiff instances:
        except AttributeError:
            #Tries adding autodiff instance and number together

            for key in self.der:
                anew.der[key] = self.der[key]
                anew.val = other + self.val
            self.back_partial_der = 1

        #Returns new autodiff instance
        return anew



    #FUNCTION: __radd__
    #PURPOSE: Allows commutative addition.
    def __radd__(self, other):
        return self + other

    #FUNCTION: __sub__
    #PURPOSE: Subtract an autodiff instance or number from a autodiff instance, and calculate the derivatives resulting from this action.
    def __sub__(self, other):
        if isinstance(other, (int, float, autodiff, list, np.ndarray)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be subtracted.")

        #Generate a new autodiff instance copy of self
        anew = autodiff(self.name, self.val, self.der)

        anew.function=self.__sub__

        anew.lparent = self
        anew.rparent = other


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

            self.back_partial_der = 1
            other.back_partial_der = -1
        #Otherwise, if not two autodiff instances:
        except AttributeError:
            #Tries subtracting number from autodiff instance

            for key in self.der:
                anew.der[key] = self.der[key]
                anew.val = self.val - other

            self.back_partial_der = 1
        #Returns new autodiff instance
        self.back_partial_der = 1
        return anew



    #FUNCTION: __sub__
    #PURPOSE: Subtract an autodiff instance or number from a autodiff instance, and calculate the derivatives resulting from this action.
    def __rsub__(self, other):
        return (-1*self) + other


    #FUNCTION: __pow__
    #PURPOSE: Raise this autodiff instance to a number or to another autodiff instance, and calculate the derivatives resulting from this action.
    def __pow__(self, other):

        if isinstance(other, (int, float, autodiff, list, np.ndarray)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be .")

        #Generate a new autodiff instance copy of self
        anew = autodiff(self.name, self.val, self.der)
        anew.function=self.__pow__

        anew.lparent = self
        anew.rparent = other

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

            self.back_partial_der = other.val*self.val**(other.val-1)
            other.back_partial_der = (self.val**other.val)*np.log(self.val)

        #Otherwise, if not two autodiff instances:
        except AttributeError:
            #Tries adding autodiff instance and number together
            for key in self.der:
                if self.der[key].shape == (other*(self.val**(other - 1))).shape:
                    anew.der[key] = other*(self.val**(other - 1))*self.der[key]
                else:
                    anew.der[key] = np.dot(other*(self.val**(other - 1)),self.der[key])


            anew.val = self.val**other
            self.back_partial_der = other*self.val**(other-1)
        #Returns new autodiff instance
        return anew


    #FUNCTION: __rpow__
    #PURPOSE: Raise this number to an autodiff instance, and calculate the derivatives resulting from this action.
    def __rpow__(self, other):
        if isinstance(other, (int, float, autodiff)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be multiplied.")

        #Generate a new autodiff instance copy of self
        anew = autodiff(self.name, self.val, self.der)
        anew.function=self.__rpow__
        anew.lparent=self
        anew.rparent=other
        #Tries autodiff instance and number together
        for key in self.der:
            anew.der[key] = (other**self.val)*np.log(other)*self.der[key]

        anew.val = other**self.val
        self.back_partial_der = other**(self.val)*np.log(other)
        #Return new autodiff instance
        return anew

    def jacobian(self, order=None):
        """Returns a dictionary containing an ND-array representation of the derivatives of this autodiff instance, as well as the ordering of the variables that those derivatives are taken in respect to.

        INPUTS
        =======
        None

        RETURNS
        ========
        dictionary containing array representation (under key "jacobian") and the ordering of the variables (under key "order")

        EXAMPLES
        =========
        >>> from autodiffpy import autodiffmod as ad
        >>> from autodiffpy import autodiff_math as admath
        >>> x = ad.autodiff('x', 3)
        >>> y = ad.autodiff('y', 4)
        >>> f1 = admath.sqrt(x*y)
        >>> resdict = f1.jacobian(order=['y', 'x'])
        >>> print(resdict["order"], resdict["jacobian"][0], resdict['jacobian'][1])
        ['y', 'x'] [0.4330127] [0.57735027]
        """
        if order is not None: # If specific ordering requested
            order = list(order)
            jacobian = [None]*len(order)
            ii = 0 # For indexing through jacobian
            try:
                for key in order:
                    jacobian[ii] = self.der[key]
                    ii = ii + 1
            except KeyError:
                raise KeyError("Error: variable(s) in order have not been encountered by this autodiff instance.")

        else: # If no specific ordering given
            jacobian = [None]*len(self.der)
            order = [None]*len(self.der) # To hold ordering
            ii = 0 # For indexing through jacobian
            for key in self.der:
                order[ii] = key
                jacobian[ii] = self.der[key]
                ii = ii + 1

        # Cast the output as an array
        jacobian = np.asarray(jacobian)
        # Return jacobian and its ordering
        return {"jacobian":jacobian, "order":order}


    def backprop(self,y_true, loss = 'MSE', backproplist = None, loss_value = 0):
        if backproplist == None:
            if isinstance(y_true,list):
                y_true = np.asarray(y_true)
            elif isinstance(y_true, (float, int)):
                y_true = np.asarray([y_true])
            backproplist = {}

            if loss == 'MSE':
                d_loss = (2/y_true.shape[0]*(self.val-y_true))
                loss_value = (1/y_true.shape[0])*np.sum((self.val-y_true)**2)
            elif loss == 'MAE':
                d_loss = []
                for idx, yt in enumerate(y_true):
                    if self.val[idx]-yt>=0:
                        d_loss.append(1/y_true.shape[0])
                    else:
                        d_loss.append(-1/y_true.shape[0])
                d_loss = np.asarray(d_loss)
                loss_value = (1/y_true.shape[0])*np.sum(np.absolute((self.val-y_true)))
            elif loss == 'RMSE':
                d_loss = (1/y_true.shape[0])**(-0.5)*(self.val-y_true)/(np.sum((self.val-y_true)**2))
                loss_value = ((1/y_true.shape[0])*np.sum((self.val-y_true)**2))**(0.5)

            self.back_der = d_loss

        if self.lparent:

            self.lparent.back_der = self.back_der*self.lparent.back_partial_der
            self.lparent.backprop(y_true,loss,backproplist, loss_value)

        if isinstance(self.rparent, autodiff):

            self.rparent.back_der = self.back_der*self.rparent.back_partial_der
            self.rparent.backprop(y_true,loss,backproplist,loss_value)

        if self.lparent is None and self.rparent is None:
            backproplist[self.name] = self.back_der

        return (backproplist, loss_value)


    def forwardprop(self):
        if self.lparent.lparent is None:
            self.lparent.forwardpropcomplete = "Yes"
        if isinstance(self.rparent, autodiff) and self.rparent.lparent is None:
            self.rparent.forwardpropcomplete = "Yes"


        if self.lparent.forwardpropcomplete == 'No':
            self.lparent=self.lparent.forwardprop()

        if isinstance(self.rparent,autodiff):
            if self.rparent.forwardpropcomplete == 'Yes':
                self = self.function(self.rparent)

        if 'mul' in str(self.function):
            self.function = self.lparent.__mul__
        elif 'add' in str(self.function):
            self.function = self.lparent.__add__
        elif 'sub' in str(self.function):
            self.function = self.lparent.__sub__
        elif 'rtruediv' in str(self.function):
            self.function = self.lparent.__rtruediv__
        elif 'div' in str(self.function):
            self.function = self.lparent.__truediv__
        elif 'rpow' in str(self.function):
            self.function = self.lparent.__rpow__
        elif 'pow' in str(self.function):
            self.function = self.lparent.__pow__
        elif 'neg' in str(self.function):
            self.function = self.lparent.__neg__

        if (isinstance(self.rparent, (int,float,list,np.ndarray,autodiff))):
            self = self.function(self.rparent)
        else:
            self = self.function(self.lparent)


        return self


    def weight_update(self,delta,learning_rate):
        a = []
        for idx,value in enumerate(delta):
            a.append(np.sum(value))
        delta = a
        learning_rate = np.asarray(learning_rate)
        self.val = self.val-learning_rate*delta



def gradient_descent(f,y_true, loss = 'MSE', beta= 0.01, max_iter = 10000, tol=10**(-8)):
    """Runs gradient descent for the given function, using the specified loss function to calculate loss.

    INPUTS
    =======
    f: autodiff instance
    y_true: desired outputs
    loss: string name of the desired loss function; allowed types are ['MSE', 'MAE', 'RMSE']
    beta: learning rate (constant)
    max_iter: maximum allowed number of iterations
    tol: minimum desired loss for the function

    RETURNS
    ========
    dictionary containing the following keys and values:
       'f': the final function autodiff instance
       'w': the final weights
       'loss_array': an array of all losses for all iterations (under key 'loss_array')
       'max_iter': total number of iterations

    EXAMPLES
    =========
    >>> from autodiffpy import autodiffmod as ad
    >>> import numpy as np
    >>> x_data = np.linspace(1,5,5)
    >>> y_true = 3*x_data
    >>> w = ad.autodiff('w', [1, 1, 1, 1, 1])
    >>> f1 = w*x_data
    >>> g = ad.gradient_descent(f1, y_true, loss='MSE', beta=0.001, max_iter=10000, tol=1E-3)
    >>> print(g['loss_array'][-1] <= 1E-3)
    True
    """
    #get w from f
    j = 0
    w=f.lparent
    while j<100 and w.lparent is not None:
        w = w.lparent
        j += 1
    if w.name != 'w':
        raise ValueError('Could not find weight vector. Be sure to name the weight autodiff as "w"')

    loss_values = []

    i = 0
    loss_v = 1
    while i<max_iter and loss_v>tol:
        backprop_ans = f.backprop(y_true, loss = loss)
        delta = backprop_ans[0]
        loss_v = backprop_ans[1]
        loss_values.append(loss_v)
        # d = np.sum(np.absolute(delta['w']))
        l = beta
        '''if d>10**8:
            l = 10**(-6)
        elif d>10**5:
            l = 10**(-5)
        elif d>1000:
            l = 0.0001
        elif d>10:
            l = 0.001
        else:
            l=0.01'''
        w.weight_update(delta['w'],l)
        if loss_v > tol:
            f = f.forwardprop()
        i=i+1




    return {"f":f,"w":w,"loss_array":loss_values,"num_iter":i}
