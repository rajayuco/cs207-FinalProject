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


## Test logarithm types
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

## Test logarithm function, any base
def test_log_result_base():
    x = ad.autodiff('x', 16)
    y = ad.autodiff('y', 2)
    f1 = admath.log(x*y, 2)
    assert f1.val == 5.0 and f1.der == {'x': 1/(16*math.log(2, math.e)), 'y': 1/(2*math.log(2, math.e))}


## Test error for nonpositive logarithm attempt
def test_log_error_nonpositive():
    x = ad.autodiff('x', -1)
    with pytest.raises(ValueError):
        admath.log(x)


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



#####################
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


def test_arctan_value():
    t = ad.autodiff('t', 0.3)
    m = ad.autodiff('m', 0.2)
    f2 = admath.arctan(t*m)
    assert f2.val == np.arctan(0.3*0.2)
    assert f2.der == {'t': 0.19928258270227184, 'm': 0.2989238740534077}


def test_arctan_types():
    with pytest.raises(AttributeError):
        admath.arctan(12.1)
