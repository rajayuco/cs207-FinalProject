# import packages
import numpy as np
from autodiffpy import autodiff

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
        anew = autodiff.autodiff(name=ad.name, val = np.exp(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]*anew.val

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
        anew = autodiff.autodiff(name=ad.name, val = np.sin(ad.val), der = ad.der)
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
        anew = autodiff.autodiff(name=ad.name, val = np.cos(ad.val), der = ad.der)
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
        anew = autodiff.autodiff(name=ad.name, val = np.tan(ad.val), der = ad.der)
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
    if ad.val<=0:
        raise ValueError('Error: cannot evaluate the log of a nonpositive number')
    try:    
        anew = autodiff.autodiff(name = ad.name, val = np.log(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]/ad.val
        return anew
    except TypeError:
        print("Error: input should be autodiff instance")
