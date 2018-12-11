import pytest
import sys
sys.path.append('..')
from autodiffpy import autodiff as ad
from autodiffpy import autodiff_math as admath
import numpy as np
import math



## test sqrt function
def test_sqrt_result():
    x = ad.autodiff('x', 3)
    y = ad.autodiff('y', 2)
    f1 = 2*x + y
    assert admath.sqrt(f1).val == np.sqrt(f1.val)
    assert admath.sqrt(f1).der['x'] == 1/np.sqrt(f1.val)

def test_sqrt_type():
    with pytest.raises(AttributeError):
        admath.sqrt(10)

## Test error for nonpositive square root attempt
def test_sqrt_error_nonpositive():
    x = ad.autodiff('x', -1)
    with pytest.raises(ValueError):
        admath.sqrt(x)

def test_sqrt_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights

    # Set up parameters for gradient descent
    f1 = ad.sqrt(w*x)
    assert f1 == f1.forwardprop()


## Test sine function
def test_sin_result_single():
    x = ad.autodiff('x', 10)
    assert admath.sin(admath.sin(x)).val == np.sin(np.sin(10))
    assert admath.sin(admath.sin(x)).der['x'] == np.cos(10)*np.cos(np.sin(10))

def test_sin_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights

    # Set up parameters for gradient descent
    f1 = ad.sin(w*x)
    assert f1 == f1.forwardprop()

## Test cosine function
def test_cos_result_single():
    x = ad.autodiff('x', 10)
    assert admath.cos(admath.cos(x)).val == np.cos(np.cos(10))
    assert admath.cos(admath.cos(x)).der['x'] == np.sin(10)*np.sin(np.cos(10))

def test_cos_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.cos(w*x)
    assert f1 == f1.forwardprop()

## Test tangent function
def test_tan_result_single():
    x = ad.autodiff('x', 10)
    assert admath.tan(admath.tan(x)).val == np.tan(np.tan(10))
    assert admath.tan(admath.tan(x)).der['x'] == 1/np.cos(10)**2 * 1/np.cos(np.tan(10))**2

def test_tan_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.tan(w*x)
    assert f1 == f1.forwardprop()

## Test trigonometric types
def test_trig_type():

    with pytest.raises(AttributeError):
        admath.sin("green")
    with pytest.raises(AttributeError):
        admath.cos("green")
    with pytest.raises(AttributeError):
        admath.tan("green")

## Test exponential function
def test_exp_result_single():
    x = ad.autodiff('x', 10)
    assert admath.exp(x).val == np.exp(10)


## Test exponential types
def test_exp_types():
    with pytest.raises(AttributeError):
        admath.exp(1)

## Test logarithm types
def test_log_types():
    with pytest.raises(AttributeError):
        admath.log(1)


def test_exp_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.exp(w*x)
    assert f1 == f1.forwardprop()

## Test logarithm function
def test_log_result_single():
    x = ad.autodiff('x', 5)
    f = admath.log(x)
    assert f.val == 1.6094379124341003 and f.der == {'x': 0.2}

def test_log_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.log(w*x)
    assert f1 == f1.forwardprop()

## Test logarithm function, any base
def test_log_result_base():
    x = ad.autodiff('x', 16)
    y = ad.autodiff('y', 2)
    f1 = admath.log(x*y, 2)
    assert f1.val == 5.0 and f1.der == {'x': 1/(16*np.log(2)), 'y': 1/(2*np.log(2))}


## Test error for nonpositive logarithm attempt
def test_log_error_nonpositive():
    x = ad.autodiff('x', -1)
    with pytest.raises(ValueError):
        admath.log(x)


## Test hyperbolic types
def test_hyperbolic_types():
    with pytest.raises(AttributeError):
        admath.sinh(1)
    with pytest.raises(AttributeError):
        admath.cosh(1)
    with pytest.raises(AttributeError):
        admath.tanh(1)

## Test sinh function
def test_sinh_result_single():
    x = ad.autodiff('x', 10)
    assert admath.sinh(admath.sinh(x)).val == np.sinh(np.sinh(10))
    assert admath.sinh(admath.sinh(x)).der['x'] == np.cosh(10)*np.cosh(np.sinh(10))

def test_sinh_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.sinh(w*x)
    assert f1 == f1.forwardprop()


## Test cosh function
def test_cosh_result_single():
    x = ad.autodiff('x', 10)
    assert admath.cosh(admath.cosh(x)).val == np.cosh(np.cosh(10))
    assert admath.cosh(admath.cosh(x)).der['x'] == np.sinh(10)*np.sinh(np.cosh(10))

def test_cosh_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.cosh(w*x)
    assert f1 == f1.forwardprop()


## Test tanh function
def test_tanh_result_single():
    x = ad.autodiff('x', 10)
    assert admath.tanh(admath.tanh(x)).val == np.tanh(np.tanh(10))
    assert admath.tanh(admath.tanh(x)).der['x'] == ((1.0/np.cosh(10))**2)*((1.0/np.cosh(np.tanh(10)))**2)


def test_tanh_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.tanh(w*x)
    assert f1 == f1.forwardprop()


## Test logistic types
def test_logistic_types():
    x = ad.autodiff('x', 10)
    with pytest.raises(AttributeError):
        admath.logistic(1, k="w")
    with pytest.raises(TypeError):
        admath.logistic(x, A="wow", k=0, x0=1)
    with pytest.raises(TypeError):
        admath.logistic(x, A=3.0, k=None, x0=-1.0)
    with pytest.raises(TypeError):
        admath.logistic(x, A=0.0, k=0.0, x0="3")

## Test logistic function
def test_logistic_result_single():
    x = ad.autodiff('x', 10)

    assert admath.logistic(admath.logistic(x)).val == 1.0/(1.0 + np.exp(-1*(1.0/(1 + np.exp(-10)))))
    assert admath.logistic(admath.logistic(x, A=-1, k=3.5, x0=-2), A=2, k=-1, x0=5).val == 2.0/(1.0 + np.exp(-1*-1*((-1.0/(1.0 + np.exp(-1*3.5*(10 - -2)))) - 5)))

## Test logistic function
def test_logistic_result_dot():
    x = ad.autodiff('x', 10)


    assert admath.logistic(admath.logistic(x)).val == 1.0/(1.0 + np.exp(-1*(1.0/(1 + np.exp(-10)))))
    assert admath.logistic(admath.logistic(x, A=-1, k=3.5, x0=-2), A=2, k=-1, x0=5).val == 2.0/(1.0 + np.exp(-1*-1*((-1.0/(1.0 + np.exp(-1*3.5*(10 - -2)))) - 5)))

def test_logistic_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.logistic(w*x, A=2.0, k=1.5, x0=0.7)
    assert f1 != f1.forwardprop()


def test_arcsin_value():
    t = ad.autodiff('t', 0.3)
    m = ad.autodiff('m', 0.2)
    f2 = admath.arcsin(t*m)
    assert f2.val == np.arcsin(0.3*0.2)
    assert f2.der == {'t': 0.2003609749252153, 'm': 0.3005414623878229}

def test_arcsin_types():
    with pytest.raises(AttributeError):
        admath.arcsin(12.1)

## Test error for nonpositive arcsin attempt
def test_arcsin_error_nonpositive():
    x = ad.autodiff('x', 2)
    with pytest.raises(ValueError):
        admath.arcsin(x)

def test_arcsin_dotproduct():
    x = np.array([[0.1,-0.2,0.1],[0.3,0,0.4]]) #Data
    w = ad.autodiff('w', [0.1,0.1,0.1]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.arcsin(w*x)
    assert f1 == f1.forwardprop()


def test_arccos_value():
    t = ad.autodiff('t', 0.3)
    m = ad.autodiff('m', 0.2)
    f2 = admath.arccos(t*m)
    assert f2.val == np.arccos(0.3*0.2)
    assert f2.der == {'t':  -0.2003609749252153, 'm': -0.3005414623878229}


def test_arccos_types():
    with pytest.raises(AttributeError):
        admath.arccos(12.1)


## Test error for nonpositive arccos attempt
def test_arccos_error_nonpositive():
    x = ad.autodiff('x', 2)
    with pytest.raises(ValueError):
        admath.arccos(x)

def test_arccos_dotproduct():
    x = np.array([[0.1,-0.2,0.1],[0.3,0,0.4]]) #Data
    w = ad.autodiff('w', [0.1,0.1,0.1]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.arccos(w*x)
    assert f1 == f1.forwardprop()

def test_arctan_value():
    t = ad.autodiff('t', 0.3)
    m = ad.autodiff('m', 0.2)
    f2 = admath.arctan(t*m)
    assert f2.val == np.arctan(0.3*0.2)
    assert f2.der == {'t': 0.19928258270227184, 'm': 0.2989238740534077}


def test_arctan_types():
    with pytest.raises(AttributeError):
        admath.arctan(12.1)

def test_arctan_dotproduct():
    x = np.array([[1,-2,1],[3,0,4]]) #Data
    w = ad.autodiff('w', [3,-1,0]) #Weights
    # Set up parameters for gradient descent
    f1 = ad.arctan(w*x)
    assert f1 == f1.forwardprop()
