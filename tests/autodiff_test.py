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
def test_truediv_ad_result():
    f1 = x/y
    assert f1.val == 5
    assert f1.der['x'] == 1/2
    assert f1.der['y'] == 10/4

def test_truediv_const_result():
    f2 = x/3
    assert f2.val == 10/3
    assert f2.der['x'] == 1/3

def test_rdiv_ad_result():
    f1 = y/x
    assert f1.val == 2/10
    assert f1.der['x'] == 2/10**2
    assert f1.der['y'] == 1/10
    
def test_exp_result():
    assert x.exp().val == np.exp(10)
	

## test multiplication
def test_mul_ad_result():
    f1 = x*y
    assert f1.val == 20
    assert f1.der['x'] == 2
    assert f1.der['y'] == 10

def test_mul_const_result():
    f2 = 3*x
    assert f2.val == 30
    assert f2.der['x'] == 3

def test_mul_types():
    s = "str"
    with pytest.raises(TypeError):
        x*s

## test unary function (negative)
def test_neg_result():
    assert -x.val == -10
    assert -x.der['x'] == -1
