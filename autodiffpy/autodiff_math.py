# import packages
import numpy as np
import sys
sys.path.append('..')
try:
    import autodiff
except:
    from autodiffpy import autodiff




def sqrt(ad):
    """Returns autodiff instance of sqrt(x)

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
    >>> x = autodiff.autodiff('x', 3)
    >>> y = autodiff.autodiff('y', 4)
    >>> f1 = admath.sqrt(x*y)
    >>> print(f1.val, f1.der)
    [3.46410162] {'x': [0.57735027], 'y': [0.4330127]}
    """
    try:
        # Check that the domain of the square root is valid
        if np.min(ad.val) < 0:
            raise ValueError('Error: cannot evaluate the square root of a negative number(s).')
        
        # Create a new autodiff instance with forward result
        anew = autodiff.autodiff(name = ad.name, val = np.sqrt(ad.val), der = ad.der)
        for key in ad.der: # Calculate derivatives for each variable
            anew.der[key] = 1/(2*np.sqrt(ad.val))*ad.der[key]
        
        # Update with the backpropagation derivatives
        anew.lparent = ad
        ad.back_partial_der = (1/2.0)*((ad.val)**(-1/2.0))
        return anew
    except AttributeError: #If non-autodiff instance passed
        raise AttributeError("Error: input should be autodiff instance only.")


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
    [-0.54402111] {'x': array([-0.83907153])}
    """
    try:
        anew = autodiff.autodiff(name=ad.name, val = np.sin(ad.val), der = ad.der)
        anew.lparent = ad
        for key in ad.der:
            anew.der[key] = np.cos(ad.val)*ad.der[key]
        ad.back_partial_der = np.cos(ad.val)
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
    [-0.83907153] {'x': array([0.54402111])}
    """
    try:
        anew = autodiff.autodiff(name=ad.name, val = np.cos(ad.val), der = ad.der)
        anew.lparent = ad
        for key in ad.der:
            anew.der[key] = -np.sin(ad.val)*ad.der[key]
        ad.back_partial_der = -np.sin(ad.val)
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
    [0.64836083] {'x': array([1.42037176])}
    """
    try:
        anew = autodiff.autodiff(name=ad.name, val = np.tan(ad.val), der = ad.der)
        anew.lparent = ad
        for key in ad.der:
            anew.der[key] = 1/(np.cos(ad.val))**2*ad.der[key]
        ad.back_partial_der = 1/(np.cos(ad.val))**2
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

def log(ad, base = np.e):

    '''Returns autodiff instance of log(x)

    INPUTS
    ==========
    ad: autodiff instance
    base: base of the log. By default, log(x) is natural log.

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
    [2.]
    >>> print(f1.der['x'])
    [0.13533528]
    '''
    try:
        if isinstance(ad.val, list):
            for val in ad.val:
                if val <= 0:
                    raise ValueError('Error: cannot evaluate the log of a nonpositive number.')

        elif ad.val<=0:
                raise ValueError('Error: cannot evaluate the log of a nonpositive number.')

        anew = autodiff.autodiff(name = ad.name, val = np.log(ad.val)/np.log(base), der = ad.der)
        anew.lparent = ad
        for key in ad.der:
            anew.der[key] = ad.der[key]/(ad.val*(np.log(base)/np.log(np.e)))
        ad.back_partial_der = 1/(ad.val*(np.log(base)/np.log(np.e)))
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
    >>> print(f1.val == [np.exp(10)])
    [ True]
    >>> print(f1.der['x'] == [np.exp(10)])
    [ True]
    '''

    try:
        anew = autodiff.autodiff(name=ad.name, val = np.exp(ad.val), der = ad.der)
        anew.lparent = ad
        for key in ad.der:
            anew.der[key] = ad.der[key]*anew.val
        ad.back_partial_der = anew.val
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")


def arcsin(ad):
    """Returns autodiff instance of arcsin(x)

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
    >>> x = autodiff.autodiff('x', 0.1)
    >>> f1 = admath.arcsin(x)
    >>> print(f1.val, f1.der)
    [0.10016742] {'x': array([1.00503782])}
    """
    try:
        if ad.val**2 > 1:
            raise ValueError('Error: invalid value encountered while calculating derivatives.')
        anew = autodiff.autodiff(name=ad.name, val = np.arcsin(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = 1/np.sqrt(1 - ad.val**2)*ad.der[key]
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")

def arccos(ad):
    """Returns autodiff instance of arccos(x)

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
    >>> x = autodiff.autodiff('x', 0.2)
    >>> f1 = admath.arccos(x)
    >>> print(f1.val, f1.der)
    [1.36943841] {'x': array([-1.02062073])}
    """
    try:
        if ad.val**2 > 1:
            raise ValueError('Error: invalid value encountered while calculating derivatives.')
        anew = autodiff.autodiff(name=ad.name, val = np.arccos(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = -1/np.sqrt(1 - ad.val**2)*ad.der[key]
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")


def arctan(ad):
    """Returns autodiff instance of arctan(x)

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
    >>> x = autodiff.autodiff('x', 0.2)
    >>> f1 = admath.arctan(x)
    >>> print(f1.val, f1.der)
    [0.19739556] {'x': array([0.96153846])}
    """
    try:
        anew = autodiff.autodiff(name=ad.name, val = np.arctan(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = 1/(1+ad.val**2)*ad.der[key]
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
    >>> print(f1.val == [np.sinh(5)])
    [ True]
    >>> print(f1.der['x'] == [np.cosh(5)])
    [ True]
    '''

    try:
        # Create a new autodiff instance with forward result
        anew = autodiff.autodiff(name=ad.name, val = np.sinh(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]*np.cosh(ad.val)
        # Update with the backpropagation derivatives
        anew.lparent = ad
        ad.back_partial_der = np.cosh(ad.val)

        return anew
    except AttributeError: #If non-autodiff instance passed
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
    >>> print(f1.val == [np.cosh(5)])
    [ True]
    >>> print(f1.der['x'] == [np.sinh(5)])
    [ True]
    '''

    try:
        # Create a new autodiff instance with forward result
        anew = autodiff.autodiff(name=ad.name, val = np.cosh(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]*np.sinh(ad.val)
        # Update with the backpropagation derivatives
        anew.lparent = ad
        ad.back_partial_der = np.sinh(ad.val)

        return anew
    except AttributeError: #If non-autodiff instance passed
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
    >>> print(f1.val == [np.tanh(5)])
    [ True]
    >>> print(f1.der['x'] == [(1.0/np.cosh(5))**2])
    [ True]
    '''

    try:
        # Create a new autodiff instance with forward result
        anew = autodiff.autodiff(name=ad.name, val = np.tanh(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = ad.der[key]*((1.0/np.cosh(ad.val))**2)
        # Update with the backpropagation derivatives
        anew.lparent = ad
        ad.back_partial_der = ((1.0/np.cosh(ad.val))**2)

        return anew
    except AttributeError: #If non-autodiff instance passed
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
    >>> print(f1.val == [testresult])
    [ True]
    >>> print(f1.der['x'] == [12*np.exp(-4*(5 - 7))/1.0/((np.exp(-4*(5 - 7)) + 1)**2)])
    [ True]
    '''

    try:
        # Create a new autodiff instance with forward result
        anew = autodiff.autodiff(name=ad.name, val = (A/1.0/(1.0 + np.exp(-1.0*k*(ad.val - x0)))), der = ad.der)
        for key in ad.der:
            anew.der[key] = (A*k*ad.der[key])*np.exp(-1.0*k*(ad.val - x0))/1.0/((np.exp(-1.0*k*(ad.val - x0)) + 1.0)**2)
        # Update with the backpropagation derivatives
        anew.lparent = ad
        ad.back_partial_der = (A*k)*np.exp(-1.0*k*(ad.val - x0))/1.0/((np.exp(-1.0*k*(ad.val - x0)) + 1.0)**2)

        return anew
    except AttributeError: #If non-autodiff instance passed
        raise AttributeError("Error: input should be autodiff instance only.")
    except TypeError:
        raise TypeError("Error: input attributes A, k, and x0 should be numbers.")
