import pytest
import sys
sys.path.append('..')
from autodiffpy import autodiff as ad
from autodiffpy import autodiff_math as admath
import numpy as np



## Test sine function
def test_sin_result_single():
    x = ad.autodiff('x', 10)
    assert admath.sin(admath.sin(x)).val == np.sin(np.sin(10))
    assert admath.sin(admath.sin(x)).der['x'] == np.cos(10)*np.cos(np.sin(10))

## Test cosine function
def test_cos_result_single():
    x = ad.autodiff('x', 10)
    assert admath.cos(admath.cos(x)).val == np.cos(np.cos(10))
    assert admath.cos(admath.cos(x)).der['x'] == np.sin(10)*np.sin(np.cos(10))

## Test tangent function
def test_tan_result_single():
    x = ad.autodiff('x', 10)
    assert admath.tan(admath.tan(x)).val == np.tan(np.tan(10))
    assert admath.tan(admath.tan(x)).der['x'] == 1/np.cos(10)**2 * 1/np.cos(np.tan(10))**2

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

## Test logarithm function
def test_log_result_single():
    x = ad.autodiff('x', 5, 2)
    f = admath.log(x)
    assert f.val == 1.6094379124341003 and f.der == {'x': 0.4}

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

## Test cosh function
def test_cosh_result_single():
    x = ad.autodiff('x', 10)
    assert admath.cosh(admath.cosh(x)).val == np.cosh(np.cosh(10))
    assert admath.cosh(admath.cosh(x)).der['x'] == np.sinh(10)*np.sinh(np.cosh(10))

## Test tanh function
def test_tanh_result_single():
    x = ad.autodiff('x', 10)
    assert admath.tanh(admath.tanh(x)).val == np.tanh(np.tanh(10))
    assert admath.tanh(admath.tanh(x)).der['x'] == ((1.0/np.cosh(10))**2)*((1.0/np.cosh(np.tanh(10)))**2)


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

