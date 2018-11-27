# import packages
import numpy as np
import sys
sys.path.append('..')
try:
    import autodiff
except:
    from autodiffpy import autodiff


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
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 10)
    >>> f1 = admath.sin(x)
    >>> print(f1.val, f1.der)
    -0.5440211108893698 {'x': -0.8390715290764524}
    """
    try:
        anew = autodiff.autodiff(name=ad.name, val = np.sin(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = np.cos(ad.val)*ad.der[key]
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")


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
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 10)
    >>> f1 = admath.cos(x)
    >>> print(f1.val, f1.der)
    -0.8390715290764524 {'x': 0.5440211108893698}
    """
    try:
        anew = autodiff.autodiff(name=ad.name, val = np.cos(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = -np.sin(ad.val)*ad.der[key]
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

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
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 10)
    >>> f1 = admath.tan(x)
    >>> print(f1.val, f1.der)
    0.6483608274590866 {'x': 1.4203717625834316}
    """
    try:
        anew = autodiff.autodiff(name=ad.name, val = np.tan(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = 1/(np.cos(ad.val))**2*ad.der[key]
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

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
    >>> import numpy as np
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', np.exp(2))
    >>> f1 = admath.log(x)
    >>> print(f1.val)
    2.0
    >>> print(f1.der['x'])
    0.1353352832366127
    '''


    try:
        if ad.val<=0:
            raise ValueError('Error: cannot evaluate the log of a nonpositive number')

        anew = autodiff.autodiff(name = ad.name, val = np.log(ad.val), der = ad.der)

        for key in ad.der:
            anew.der[key] = ad.der[key]/ad.val
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

def exp(ad):
    '''Returns autodiff instance of exp(x)

    INPUTS
    ==========
    ad: autodiff instance

    RETURNS
    ==========
    anew: autodiff instance with updated values and derivatives

    EXAMPLES
    ==========
    >>> import numpy as np
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 10)
    >>> f1 = admath.exp(x)
    >>> print(f1.val == np.exp(10))
    True
    >>> print(f1.der['x'] == np.exp(10))
    True
    '''

    try:
        anew = autodiff.autodiff(name=ad.name, val = np.exp(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]*anew.val

        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

def sinh(ad):
    '''Returns autodiff instance of sinh(x)

    INPUTS
    ==========
    ad: autodiff instance

    RETURNS
    ==========
    anew: autodiff instance with updated values and derivatives

    EXAMPLES
    ==========
    >>> import numpy as np
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 5)
    >>> f1 = admath.sinh(x)
    >>> print(f1.val == np.sinh(5))
    True
    >>> print(f1.der['x'] == np.cosh(5))
    True
    '''

    try:
        anew = autodiff.autodiff(name=ad.name, val = np.sinh(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]*np.cosh(ad.val)

        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

def cosh(ad):
    '''Returns autodiff instance of cosh(x)

    INPUTS
    ==========
    ad: autodiff instance

    RETURNS
    ==========
    anew: autodiff instance with updated values and derivatives

    EXAMPLES
    ==========
    >>> import numpy as np
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 5)
    >>> f1 = admath.cosh(x)
    >>> print(f1.val == np.cosh(5))
    True
    >>> print(f1.der['x'] == np.sinh(5))
    True
    '''

    try:
        anew = autodiff.autodiff(name=ad.name, val = np.cosh(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]*np.sinh(ad.val)

        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

def tanh(ad):
    '''Returns autodiff instance of tanh(x)

    INPUTS
    ==========
    ad: autodiff instance

    RETURNS
    ==========
    anew: autodiff instance with updated values and derivatives

    EXAMPLES
    ==========
    >>> import numpy as np
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 5)
    >>> f1 = admath.tanh(x)
    >>> print(f1.val == np.tanh(5))
    True
    >>> print(f1.der['x'] == (np.sech(5))**2)
    True
    '''

    try:
        anew = autodiff.autodiff(name=ad.name, val = np.tanh(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]*((np.sech(ad.val))**2)

        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

def logistic(ad, A=1.0, k=1.0, x0=0.0):
    '''Returns autodiff instance of the logistic function of x

    INPUTS
    ==========
    ad: autodiff instance
    A: maximum value of this logistic function
    k: growth rate (steepness) of the logistic function
    x0: x-axis location of the logistic function's midpoint

    RETURNS
    ==========
    anew: autodiff instance with updated values and derivatives

    EXAMPLES
    ==========
    >>> import numpy as np
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 5)
    >>> f1 = admath.logistic(x, A=3, k=4, x0=7)
    >>> testresult = 3.0/(1 + np.exp(-4*(5-7)))
    >>> print(f1.val == testresult)
    True
    >>> print(f1.der['x'] == testresult*(1 - testresult))
    True
    '''

    try:
        anew = autodiff.autodiff(name=ad.name, val = (A/1.0/(1.0 + np.exp(-1.0*k*(ad.val - x0)))), der = ad.der)
        for key in ad.der:
            anew.der[key] = (A*k*ad.der[key])*np.exp(-1.0*k*(ad.val - x0))/1.0/((np.exp(-1.0*k*(ad.val - x0)) + 1.0)**2)

        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")
    except TypeError:
        raise TypeError("Error: input attributes A, k, and x0 should be numbers.")



#