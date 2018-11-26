# import packages
import numpy as np
import math
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
    3.4641016151377544 {'x': 0.5773502691896258, 'y': 0.4330127018922194}
    """
    try:
        if ad.val<0:
            raise ValueError('Error: cannot evaluate the square root of a negative number')

        anew = autodiff.autodiff(name = ad.name, val = np.sqrt(ad.val), der = ad.der)
        anew.lparent = ad
        for key in ad.der:
            anew.der[key] = 1/(2*np.sqrt(ad.val))*ad.der[key]
        ad.back_partial_der = anew.val
        return anew
    except AttributeError:
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

def log(ad, base = math.e):

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
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', np.exp(2))
    >>> f1 = admath.log(x)
    >>> f1.val = 2.0
    >>> f1.der = 0.1353352832366127
    '''


    try:
        if ad.val<=0:
            raise ValueError('Error: cannot evaluate the log of a nonpositive number')

        anew = autodiff.autodiff(name = ad.name, val = math.log(ad.val, base), der = ad.der)

        for key in ad.der:
            anew.der[key] = ad.der[key]/(ad.val*math.log(base, math.e))
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
    >>> from autodiffpy import autodiff
    >>> from autodiffpy import autodiff_math as admath
    >>> x = autodiff.autodiff('x', 10)
    >>> f1 = admath.exp(x)
    >>> f1.val = np.exp(10)
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
    0.1001674211615598 {'x': 1.005037815259212}
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
    1.369438406004566 {'x': -1.0206207261596576}
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
     0.19739555984988078 {'x': 0.9615384615384615}
    """
    try:
        anew = autodiff.autodiff(name=ad.name, val = np.arctan(ad.val), der = ad.der)
        for key in ad.der:
            anew.der[key] = 1/(1+ad.val**2)*ad.der[key]
        return anew
    except AttributeError:
        raise AttributeError("Error: input should be autodiff instance only.")
