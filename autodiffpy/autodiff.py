#FILE: autodiff.py


# Import packages
import numpy as np
import pandas as pd
try:
    from autodiff_math import *
except ImportError:
    from autodiffpy.autodiff_math import *


#CLASS: autodiff
#PURPOSE: Create a variable instance for performing automatic differentiation.
class autodiff():
    #FUNCTION: __init__
    #PURPOSE: Initialize the instance.
    def __init__(self,name,val,der=1):
        self.name = name # Name of the variable
        # Set val attribute to hold value(s)
        if isinstance(val, np.ndarray):
            self.val = val
        elif isinstance(val, list):
            self.val = np.asarray(val)
        else:
            self.val = np.asarray([val])

        # Set der attribute to hold derivative(s)
        if isinstance(der, np.ndarray):
            self.der = {name:der}
        elif isinstance(der, list):
            self.der = {name:np.asarray(der)}
        else:
            self.der = {name:np.asarray([der]*self.val.shape[0])}

        # Set attributes for forward and back propagation
        self.lparent = None
        self.rparent = None
        self.forwardpropcomplete = 'No'
        self.function = None
        #
        self.back_der = None
        self.back_partial_der = None


    #FUNCTION: __str__
    #PURPOSE: Print a string representation of the instance.
    def __str__(self):
       return f"value: {self.val}\nderivatives:{self.der}"


    #FUNCTION: __eq__
    #PURPOSE: Compare two autodiff instances for equality.
    def __eq__(self, other):
           # Check if both autodiff instances
           if isinstance(other, autodiff) == False:
               raise ValueError("Error: only autodiff instances can be compared with another.")
           
           # Check if equal value(s)
           if (False in (self.val == other.val)):
               return False
           
           # Check if equal derivative keys
           if (False in (np.unique([key for key in self.der]) == np.unique([key for key in other.der]))):
               return False
           
           # Check if equal derivative value(s)
           for key in self.der:
               if (False in (self.der[key] == other.der[key])):
                   return False
           return True


    #FUNCTION: __ne__
    #PURPOSE: Compare two autodiff instances for non-equality.
    def __ne__(self, other):
        return not (self == other)


    #FUNCTION: __neg__
    #PURPOSE: Return the negation of the instance.
    def __neg__(self,other=-1):
        """Allows unary operation of autodiff instance."""
        # Create a new autodiff instance with forward result
        anew = autodiff(self.name, -self.val, self.der)
        
        # Calculate derivatives for each variable
        for key in self.der:
            anew.der[key] = -1*self.der[key]

        # Update with the backpropagation derivatives
        anew.lparent = self
        anew.function = self.__neg__
        self.back_partial_der = -1
        return anew


    #FUNCTION: __mul__
    #PURPOSE: Return the multiplication of the instance and another instance/value(s).
    def __mul__(self, other):
        """Allows multiplication of another autodiff instance, or multiplication of a constant (integer or float)."""
        # Cast inputs to numpy arrays, if able
        if isinstance(other, pd.DataFrame):
            other = np.asarray(other)
        if isinstance(other, (int, float, autodiff, list, np.ndarray)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be multiplied.")
        if isinstance(other, list):
            other = np.asarray(other)

        # Create a new autodiff instance with forward result
        anew = autodiff(self.name, self.val, self.der)

        # Store for backpropagation functionality
        anew.lparent = self
        anew.rparent = other
        anew.function = self.__mul__

        # Assuming that other is autodiff instance, perform operation
        try:
            anew.val = self.val*other.val
            # Calculate derivatives for each variable
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key]=self.val*(other.der[key])
                elif key not in other.der:
                    anew.der[key]=(self.der[key])*other.val
                else:
                    anew.der[key]=(self.der[key])*other.val+(other.der[key])*self.val

            # Set the back partial derivatives that can be used for backpropagation
            self.back_partial_der = other.val
            other.back_partial_der = self.val

        # If 'other' is not autodiff instance
        except AttributeError:
            # Assuming that 'other' is a valid constant, perform operation
            if isinstance(other, (int,float)):
                anew.val = self.val*other
                # Calculate derivatives for each variable
                for key in self.der:
                    anew.der[key] = other*self.der[key]
                self.back_partial_der = other

            # If 'other' consists of several constants
            else:
                other = np.asarray(other)
                if other.shape!=anew.val.shape: # Perform a dot-product calculation
                    anew.val = np.dot(other, self.val)
                else: # Perform an element-wise calculation
                    anew.val = self.val*other
                # Calculate derivatives for each variable
                for key in self.der:
                    anew.der[key] = other
                # Determine partial derivatives
                try: # If 'other' contains several values
                    fder = [[] for i in range(len(other[0]))]
                    for idx,value in enumerate(other):
                        for idx2,value2 in enumerate(value):
                            fder[idx2].append(value2)
                except:
                    fder = other

                # Set partial derivative
                self.back_partial_der = fder
        return anew


    #FUNCTION: __rmul__
    #PURPOSE: Allow commutative multiplication.
    __rmul__ = __mul__


    #FUNCTION: __truediv__
    #PURPOSE: Return the true (left) division of the instance and another instance/value(s).
    def __truediv__(self,other):
        '''Allow true (left) division between this instance and another instance/value(s).'''
        # Cast inputs to numpy arrays, if able
        if isinstance(other, (int, float, list, np.ndarray, autodiff)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be divided.")
        if isinstance(other,list):
            other = np.asarray(other)
        
        # Create a new autodiff instance with forward result
        anew = autodiff(self.name, self.val, self.der)

        # Store for backpropagation functionality
        anew.lparent = self
        anew.rparent = other
        anew.function = self.__truediv__

        # Assuming that other is autodiff instance, perform operation
        try:
            anew.val = self.val/other.val # Calculate value
            # Store partial derivatives
            self.back_partial_der = 1/other.val
            other.back_partial_der = -self.val/(other.val**2)

            # Calculate derivatives for each variable
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key]= -self.val*other.der[key]/(other.val**2)
                elif key not in other.der:
                    anew.der[key]=self.der[key]/other.val
                else:
                    anew.der[key] = 0

        # If 'other' is not an autodiff instance
        except AttributeError:
            anew.val = self.val/other # Calculate value
            self.back_partial_der = 1/other
            
            # Calculate derivatives for each variable
            for key in self.der:
                anew.der[key] = (self.der[key])/other
        return anew


    #FUNCTION: __rtruediv__
    #PURPOSE: Return the true (right) division of the instance and another instance/value(s).
    def __rtruediv__(self, other):
        '''Allow true (right) division between this instance and another instance/value(s).'''

        # Cast inputs to numpy arrays, if able
        if isinstance(other, (int, float, list, np.ndarray, autodiff)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be divided.")
        if isinstance(other,list):
            other = np.asarray(other)

        # Store for backpropagation functionality
        anew = autodiff(self.name, self.val, self.der)
        anew.lparent = self
        anew.rparent = other
        anew.function=self.__rtruediv__

        # Calculate derivatives for each variable
        if isinstance(other, (int,float,list,np.ndarray)):
            for key in self.der:
                anew.der[key] = -other*(self.der[key])/self.val**2
                anew.val = other/self.val
                self.back_partial_der = -1*(self.val**2)
            return anew
        else:
            raise ValueError("Error: invalid input given to true (right) division.")


    #FUNCTION: __add__
    #PURPOSE: Return the addition of the instance and another instance/value(s).
    def __add__(self, other):
        '''Allow addition between this instance and another instance/value(s).'''
        # Check 'other' has an allowed type
        if isinstance(other, (int, float, autodiff, list, np.ndarray)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be added.")

        # Create a new autodiff instance with forward result
        anew = autodiff(self.name, self.val, self.der)

        # Store for backpropagation functionality
        anew.function=self.__add__
        anew.lparent = self
        anew.rparent = other

        # Assuming that other is autodiff instance, perform operation
        try:
            anew.val = self.val + other.val # Calculate value

            # Calculate derivatives for each variable
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key] = other.der[key]
                elif key not in other.der:
                    anew.der[key] = self.der[key]
               else:
                    anew.der[key] = self.der[key] + other.der[key]

            # Store partial derivatives
            self.back_partial_der = 1
            other.back_partial_der = 1

        # If 'other' is not an autodiff instance
        except AttributeError:
            # Calculate derivatives for each variable
            for key in self.der:
                anew.der[key] = self.der[key]
                anew.val = other + self.val
            # Store partial derivatives
            self.back_partial_der = 1
        return anew


    #FUNCTION: __radd__
    #PURPOSE: Allows commutative addition.
    def __radd__(self, other):
        '''Allow commutative addition.'''
        return self + other


    #FUNCTION: __sub__
    #PURPOSE: Subtract an autodiff instance or number from a autodiff instance, and calculate the derivatives resulting from this action.
    def __sub__(self, other):
        '''Allow subtraction between this instance and another instance/value(s).'''
        # Check 'other' has an allowed type
        if isinstance(other, (int, float, autodiff, list, np.ndarray)) == False:
            raise ValueError("Error: Only integer, float, list, numpy arrays, or autodiff instances can be subtracted.")

        # Create a new autodiff instance with forward result
        anew = autodiff(self.name, self.val, self.der)

        # Store for backpropagation functionality
        anew.function=self.__sub__
        anew.lparent = self
        anew.rparent = other

        # Assuming that other is autodiff instance, perform operation
        try:
            anew.val = self.val - other.val # Calculate value
            # Calculate derivatives for each variable
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key] = -1*other.der[key]
                elif key not in other.der:
                    anew.der[key] = self.der[key]
                else:
                    anew.der[key] = self.der[key] - other.der[key]

            # Store partial derivatives
            self.back_partial_der = 1
            other.back_partial_der = -1
        # If 'other' is not an autodiff instance
        except AttributeError:
            # Calculate derivatives for each variable
            for key in self.der:
                anew.der[key] = self.der[key]
                anew.val = self.val - other

            # Store partial derivatives
            self.back_partial_der = 1
        #Returns new autodiff instance
        return anew


    #FUNCTION: __rsub__
    #PURPOSE: Allow reverse subtraction.
    def __rsub__(self, other):
        '''Allow reverse subtraction.'''
        return (-1*self) + other


    #FUNCTION: __pow__
    #PURPOSE: Raise this autodiff instance to a number or to another autodiff instance, and calculate the derivatives resulting from this action.
    def __pow__(self, other):
        '''Allow exponentiation between this instance and another instance/value(s).'''
        # Check 'other' has an allowed type
        if isinstance(other, (int, float, autodiff, list, np.ndarray)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be exponentiated.")

        # Create a new autodiff instance with forward result
        anew = autodiff(self.name, self.val, self.der)

        # Store for backpropagation functionality
        anew.function=self.__pow__
        anew.lparent = self
        anew.rparent = other

        # Assuming that other is autodiff instance, perform operation
        try:
            anew.val = self.val**other.val # Calculate value
            # Calculate derivatives for each variable
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key] = anew.val*(np.log(self.val)*other.der[key])
                elif key not in other.der:
                    anew.der[key] = anew.val*(other.val*self.der[key]/1.0/self.val)
                else:
                    anew.der[key] = anew.val*((np.log(self.val)*other.der[key]) + (other.val*self.der[key]/1.0/self.val))

            # Stores partial derivatives
            self.back_partial_der = other.val*self.val**(other.val-1)
            other.back_partial_der = (self.val**other.val)*np.log(self.val)

        # If 'other' is not an autodiff instance
        except AttributeError:
            # Calculate derivatives for each variable
            for key in self.der:
                anew.der[key] = other*(self.val**(other - 1))*self.der[key]
                anew.val = self.val**other
            
            # Store partial derivatives
            self.back_partial_der = other*self.val**(other-1)
        return anew


    #FUNCTION: __rpow__
    #PURPOSE: Raise this number to an autodiff instance, and calculate the derivatives resulting from this action.
    def __rpow__(self, other):
        '''Allow reverse exponentiation between this number(s) and an autodiff instance.'''
        # Check 'other' has an allowed type
        if isinstance(other, (int, float, autodiff)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be multiplied.")

        # Create a new autodiff instance with forward result
        anew = autodiff(self.name, self.val, self.der)

        # Store for backpropagation functionality
        anew.function=self.__rpow__
        anew.lparent=self
        anew.rparent=other
        
        # Assuming that other is autodiff instance, perform operation
        for key in self.der:
            anew.der[key] = (other**self.val)*np.log(other)*self.der[key]
            anew.val = other**self.val
        self.back_partial_der = other**(self.val)*np.log(other)
        #Return new autodiff instance
        return anew


    #FUNCTION: jacobian
    #PURPOSE: Returns a dictionary containing an ND-array representation of the derivatives of the given autodiff instance.
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
        >>> from autodiffpy import autodiff
        >>> x = autodiff.autodiff('x', 3)
        >>> y = autodiff.autodiff('y', 4)
        >>> f1 = admath.sqrt(x*y)
        >>> resdict = f1.jacobian(order=['y', 'x'])
        >>> print(resdict["order"], resdict["jacobian"])
        ['y', 'x'] array([0.4330127, 0.57735027])
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


    #FUNCTION: backprop
    #PURPOSE: Performs backpropagation on this autodiff instance.
    #NOTE: This function is not intended for use by users!
    def backprop(self,y_true, loss = 'MSE', backproplist = None, loss_value = 0):
        # For no prior backpropagation
        if backproplist == None:
            # Cast inputs to numpy arrays, if able
            if isinstance(y_true,list):
                y_true = np.asarray(y_true)
            elif not isinstance(y_true, np.ndarray):
                y_true = np.asarray([y_true])
            backproplist = {}

            # Calculate loss
            # For MSE loss
            if loss == 'MSE':
                d_loss = (2/y_true.shape[0]*(self.val-y_true))
                loss_value = (1/y_true.shape[0])*np.sum((self.val-y_true)**2)
            # For MAE loss
            elif loss == 'MAE':
                d_loss = []
                for idx, yt in enumerate(y_true):
                    if self.val[idx]-yt>=0:
                        d_loss.append(1/y_true.shape[0])
                    else:
                        d_loss.append(-1/y_true.shape[0])
                d_loss = np.asarray(d_loss)
                loss_value = (1/y_true.shape[0])*np.sum(np.absolute((self.val-y_true)))
            # For RMSE loss
            elif loss == 'RMSE':
                d_loss = (1/y_true.shape[0])**(-0.5)*(self.val-y_true)/(np.sum((self.val-y_true)**2))
                loss_value = ((1/y_true.shape[0])*np.sum((self.val-y_true)**2))**(0.5)

            # Record derivative of the loss
            self.back_der = d_loss

        # If prior operation with left parent
        if self.lparent:
            self.lparent.back_der = self.back_der*self.lparent.back_partial_der
            self.lparent.backprop(y_true,loss,backproplist, loss_value)

        # If prior operation with right parent that was an autodiff instance
        if isinstance(self.rparent, autodiff):
            self.rparent.back_der = self.back_der*self.rparent.back_partial_der
            self.rparent.backprop(y_true,loss,backproplist,loss_value)

        # If no prior operation
        if self.lparent is None and self.rparent is None:
            backproplist[self.name] = self.back_der

        # Return the backpropagation and all loss values
        return (backproplist, loss_value)


    #FUNCTION: forwardprop
    #PURPOSE: Performs forward propagation on this autodiff instance.
    #NOTE: This function is not intended for use by users!
    def forwardprop(self):
        # Below deals with edge cases that represent different operations that might have produced this autodiff instance
        if self.lparent.lparent is None:
            self.lparent.forwardpropcomplete = "Yes"
        if isinstance(self.rparent, autodiff) and self.rparent.lparent is None:
            self.rparent.forwardpropcomplete = "Yes"

        if self.lparent.forwardpropcomplete == 'No':
            self.lparent=self.lparent.forwardprop()

        if isinstance(self.rparent,autodiff):
            if self.rparent.forwardpropcomplete == 'Yes':
                self = self.function(self.rparent)

        # Record the elementary function that led to this autodiff instance
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

        # Update the current instance recursively
        if (isinstance(self.rparent, (int,float,list,np.ndarray,autodiff))):
            self = self.function(self.rparent)
        else:
            self = self.function(self.lparent)

        # Return the current state
        return self


    #FUNCTION: weight_update
    #PURPOSE: Performs an update of this weight's values.
    #NOTE: This function is not intended for use by users!
    def weight_update(self,delta,learning_rate):
        a = []
        for idx,value in enumerate(delta):
            a.append(np.sum(value))
        delta = a # Change in weights
        learning_rate = np.asarray(learning_rate) # Learning rate
        # Calculate the new weights
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
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> import numpy as np
    >>> x_data = np.linspace([1, 2, 3, 4, 5])
    >>> y_true = 3*x_data
    >>> w = autodiff.autodiff('w', [1, 1, 1, 1, 1])
    >>> f1 = w*x
    >>> g = ad.gradient_descent(f1, y_true, loss='MSE', beta=0.001, max_iter=5000, tol=1E-5)
    >>> print(np.abs(g.val - y_true) <= 1E-5)
    array([True, True, True, True, True])
    """
    # Extract the weights from the function
    j = 0
    w=f.lparent
    while j<100 and w.lparent is not None:
        w = w.lparent
    # Ensure the weights are labeled as 'w'
    if w.name != 'w':
        raise ValueError('Could not find weight vector. Be sure to name the weight autodiff as "w"')

    loss_values = []
    i = 0
    loss_v = 1
    # Iteratively calculate the loss and update the weights
    while i<max_iter and loss_v>tol:
        # Calculate the loss and new change in weights
        backprop_ans = f.backprop(y_true, loss = loss)
        delta = backprop_ans[0]
        loss_v = backprop_ans[1]
        loss_values.append(loss_v)
        
        l = beta # Constant learning rate
        w.weight_update(delta['w'],l) # Update the weights
        
        # Move function forward, if tolerance has been exceeded
        if loss_v > tol:
            f = f.forwardprop()
        i=i+1 # Increment the iteration count

    # Return the function, weights, loss values, and iteration count
    return {"f":f,"w":w,"loss_array":loss_values,"num_iter":i}


