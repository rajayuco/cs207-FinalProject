import pytest
import autodiff_math as admath
import autodiff as ad

## example cases
x = ad.autodiff('x', 10)
y = ad.autodiff('y', 2)

## test trigonometric functions

def test_sin_result():
    assert admath.sin(admath.sin(x)).val == np.sin(np.sin(10))
    assert admath.sin(admath.sin(x)).der['x'] == np.cos(10)*np.cos(np.sin(10))

def test_cos_result():
    assert admath.cos(admath.cos(x)).val == np.cos(np.cos(10))
    assert admath.cos(admath.cos(x)).der['x'] == np.sin(10)*np.sin(np.cos(10))

def test_tan_result():
    assert admath.tan(admath.tan(x)).val == np.tan(np.tan(10))
    assert admath.tan(admath.tan(x)).der['x'] == 1/np.cos(10)**2 * 1/np.cos(np.tan(10))**2

def test_trig_types():
    with pytest.raises(TypeError):
        admath.sin("green")
    with pytest.raises(TypeError):
        admath.cos("green")
    with pytest.raises(TypeError):
        admath.tan("green")

## test division and exponential
def test_exp_result():
    assert admath.exp(x).val == np.exp(10)
	

#test logarithm function
def test_logarithm():
    x = ad.autodiff('x',5,2)
    f = log(x)
    assert f.val == 1.6094379124341003 and f.der == {'x': 0.4}

def test_log_nonpositive():
    x = ad.autodiff('x',-1)
    with pytest.raises(ValueError):
        np.log(-1)

