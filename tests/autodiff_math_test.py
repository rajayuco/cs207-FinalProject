import pytest
import sys
sys.path.append("..")
import autodiffpy.autodiff as ad
import autodiffpy.autodiff_math as admath

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
    assert ad.tan(admath.tan(x)).val == np.tan(np.tan(10))
    assert ad.tan(admath.tan(x)).der['x'] == 1/np.cos(10)**2 * 1/np.cos(np.tan(10))**2

## Test trigonometric types
def test_trig_type():

    with pytest.raises(TypeError):
        admath.sin("green")
    with pytest.raises(TypeError):
        admath.cos("green")
    with pytest.raises(TypeError):
        admath.tan("green")

## Test exponential function
def test_exp_result_single():
    x = ad.autodiff('x', 10)
    assert admath.exp(x).val == np.exp(10)

## test division and exponential
def test_exp_result():
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
        np.log(-1)
