import pytest
import autodiffpy as ad

## example cases
x = ad.autodiff('x', 10)
y = ad.autodiff('y', 2)

## test trigonometric functions

def test_sin_result():
    assert ad.sin(ad.sin(x)).val == np.sin(np.sin(10))
    assert ad.sin(ad.sin(x)).der['x'] == np.cos(10)*np.cos(np.sin(10))

def test_cos_result():
    assert ad.cos(ad.cos(x)).val == np.cos(np.cos(10))
    assert ad.cos(ad.cos(x)).der['x'] == np.sin(10)*np.sin(np.cos(10))

def test_tan_result():
    assert ad.tan(ad.tan(x)).val == np.tan(np.tan(10))
    assert ad.tan(ad.tan(x)).der['x'] == 1/np.cos(10)**2 * 1/np.cos(np.tan(10))**2

def test_trig_types():
    with pytest.raises(TypeError):
        ad.sin("green")
    with pytest.raises(TypeError):
        ad.cos("green")
    with pytest.raises(TypeError):
        ad.tan("green")

## test division and exponential
def test_exp_result():
    assert x.exp().val == np.exp(10)
	


