import numpy as np


class autodiff():
    def __init__(self,name,val,der=1):
        self.name = name
        self.val = val
        self.der = {name:der}
        self.lparent = None
        self.rparent = None
        self.back_der = None
        self.back_partial_der = None
        # self.func = None

    def backprop(self, backproplist = None):
        if backproplist == None:
            backproplist = []
            self.back_der = 1
        if self.lparent:
            try:
                self.lparent.back_der = self.back_der*self.lparent.back_partial_der
                self.lparent.backprop(backproplist)
            except:
                pass
        if self.rparent:
            try:
                self.rparent.back_der = self.back_der*self.rparent.back_partial_der
                self.rparent.backprop(backproplist)
            except:
                pass
        if self.lparent is None and self.rparent is None:
            backproplist.append((self.name,self.back_partial_der,self.back_der))

        return backproplist


    def __eq__(self, other):
        if isinstance(other, autodiff) == False:
            raise ValueError("Error: only autodiff instances can be compared with another.")
        return (self.val == other.val) and (self.der == other.der)

    def __ne__(self, other):
        return not (self == other)


    def __neg__(self):
        """Allows unary operation of autodiff instance."""
        anew = autodiff(self.name, -self.val, self.der)
        for key in self.der:
            anew.der[key] = -self.der[key]
        return anew



    def __mul__(self, other):
        """Allows multiplication of another autodiff instance, or multiplication of a constant (integer or float)."""

        if isinstance(other, (int, float, autodiff)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be multiplied.")

        anew = autodiff(self.name, self.val, self.der)

        #Stores for backpropagation functionality
        anew.lparent = self
        anew.rparent = other

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

            #set the back partial derivatives that can be used for backpropagation
            self.back_partial_der = other.val
            other.back_partial_der = self.val

        # if 'other' is not autodiff instance
        except AttributeError:
            # assuming that 'other' is a valid constant
            for key in self.der:
                anew.der[key] = other*self.der[key]
                anew.val = other*self.val
                self.back_partial_der = other

        return anew

    __rmul__ = __mul__

    def __truediv__(self,other):
        '''function for left division'''

        if isinstance(other, (int, float, autodiff)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be divided.")

        anew = autodiff(self.name, self.val, self.der)
        try:
            anew.val = self.val/other.val
            for key in np.unique([key for key in self.der] + [key for key in other.der]):
                if key not in self.der:
                    anew.der[key]=self.val*other.der[key]/other.val**2
                elif key not in other.der:
                    anew.der[key]=self.der[key]/other.val
                else:
                    anew.der[key]=0
        except AttributeError:
            for key in self.der:
                anew.der[key] = self.der[key]/other
                anew.val = self.val/other
        return anew


    def __rtruediv__(self, other):

        '''function for right division, when performing (constant)/(autodiff)'''

        if isinstance(other, (int,float)):
            anew = autodiff(self.name, self.val, self.der)
            for key in self.der:
                anew.der[key] = -other*self.der[key]/self.val**2
                anew.val = other/self.val
            return anew
        else:
            raise ValueError("Error: Only integer, float, or autodiff instances can be divided.")



    def __add__(self, other):

        if isinstance(other, (int, float, autodiff)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be added.")


        #Generate a new autodiff instance copy of self
        anew = autodiff(self.name, self.val, self.der)

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
        if isinstance(other, (int, float, autodiff)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be subtracted.")

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

            for key in self.der:
                anew.der[key] = self.der[key]
                anew.val = self.val - other

        #Returns new autodiff instance
        return anew



    #FUNCTION: __sub__
    #PURPOSE: Subtract an autodiff instance or number from a autodiff instance, and calculate the derivatives resulting from this action.
    def __rsub__(self, other):
        return (-1*self) + other


    #FUNCTION: __pow__
    #PURPOSE: Raise this autodiff instance to a number or to another autodiff instance, and calculate the derivatives resulting from this action.
    def __pow__(self, other):

        if isinstance(other, (int, float, autodiff)) == False:
            raise ValueError("Error: Only integer, float, or autodiff instances can be multiplied.")

        #Generate a new autodiff instance copy of self
        anew = autodiff(self.name, self.val, self.der)

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
                anew.der[key] = other*(self.val**(other - 1))*self.der[key]
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

        #Tries autodiff instance and number together
        for key in self.der:
            anew.der[key] = (other**self.val)*np.log(other)*self.der[key]
            anew.val = other**self.val

        #Return new autodiff instance
        return anew

    def jacobian(self):
        jacobian = [[],[]]
        for key in self.der:
            jacobian[0].append(key)
            jacobian[1].append(self.der[key])

        return jacobian
