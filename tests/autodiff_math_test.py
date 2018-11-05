import pytest
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
