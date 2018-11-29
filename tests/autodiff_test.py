import pytest
import sys
import numpy as np

sys.path.append('..')
from autodiffpy import autodiff as ad
from autodiffpy import autodiff_math as admath



def test_eq():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x*y
    f2 = y*x
    assert f1 == f2

def test_eq_errortypes():
    x = ad.autodiff('x', 10)
    f1 = x + 2
    f2 = "hey"
    with pytest.raises(ValueError):
        f1 == f2

def test_ne():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x*y
    f2 = y*y
    assert f1 != f2

def test_ne_errortypes():
    x = ad.autodiff('x', 10)
    y = 1
    f1 = x**2
    with pytest.raises(ValueError):
        f1 == y

## Test true division with an autodiff instsance
def test_truediv_result_ad():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x/y
    f2 = 2*x/x
    assert f1.val == 5
    assert f1.der['x'] == 1/2
    assert f1.der['y'] == 10/4
    assert f2.val == 2
    assert f2.der['x'] == 0

## Test true division with a constant
def test_truediv_result_const():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f2 = x/3
    assert f2.val == 10/3
    assert f2.der['x'] == 1/3

## Test true division error types
def test_truediv_err_types():
    x = ad.autodiff('x', 10)
    s = "str"
    with pytest.raises(ValueError):
        x/s

def test_rtruediv_err_types():
    x = ad.autodiff('x', 10)
    s = "str"
    with pytest.raises(ValueError):
        s/x

## Test reverse division with a constant
def test_rtruediv_result_const():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f2 = 3/x
    assert f2.val == 3/10
    assert f2.der['x'] == -3/100


## Test multiplication with an autodiff instance
def test_mul_result_ad():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x*y
    f2 = x*x*x
    assert f1.val == 20
    assert f1.der['x'] == 2
    assert f1.der['y'] == 10
    assert f2.val == 1000
    assert f2.der['x'] == 300

## Test multiplication with a constant
def test_mul_result_const():
    x = ad.autodiff('x', 10)
    f2 = 3*x
    assert f2.val == 30
    assert f2.der['x'] == 3

## Test multiplication error types
def test_mul_err_types():
    x = ad.autodiff('x', 10)
    s = "str"
    with pytest.raises(ValueError):
        x*s

## Test unary function (negative)
def test_neg_result_single():
    x = ad.autodiff('x', 10)
    f = -x
    assert f.val == -10
    assert f.der['x'] == -1

## Test addition with a constant and an autodiff instance
def test_add_result_adandconst():
    ad1 = ad.autodiff(name="x", val=20.5, der=1)
    ad2 = ad.autodiff(name="y", val=3, der=1)

    ad3 = ad1 + ad2 + ad1
    ad4 = ad1 + 5.5 + ad2
    ad5 = 5.5 + ad1 + ad2

    assert ad3.val == 44
    assert ad3.der["x"] == 2
    assert ad3.der["y"] == 1

    assert ad4.val == 29
    assert ad4.der["x"] == 1
    assert ad4.der["y"] == 1

    assert ad5.val == 29
    assert ad5.der["x"] == 1
    assert ad5.der["y"] == 1

## Test addition error types
def test_add_err_types():
    x = ad.autodiff('x', 10)
    s = "str"
    with pytest.raises(ValueError):
        x+s

## Test substraction error types
def test_sub_err_types():
    x = ad.autodiff('x', 10)
    s = "str"
    with pytest.raises(ValueError):
        x-s

## Test subtraction with a constant and an autodiff instance
def test_sub_result_adandconst():
    ad1 = ad.autodiff(name="x", val=2.5, der=1)
    ad2 = ad.autodiff(name="y", val=3, der=1)

    ad3 = ad1 - ad2 - ad1
    ad4 = ad1 - 5.5 - ad2
    ad5 = 5.5 - ad1 - ad2

    assert ad3.val == -3
    assert ad3.der["x"] == 0
    assert ad3.der["y"] == -1

    assert ad4.val == -6
    assert ad4.der["x"] == 1
    assert ad4.der["y"] == -1

    assert ad5.val == 0
    assert ad5.der["x"] == -1
    assert ad5.der["y"] == -1

## Test power error types
def test_pow_err_types():
    x = ad.autodiff('x', 10)
    s = "str"
    with pytest.raises(ValueError):
        x**s

## Test reverse power error types
def test_revpow_err_types():
    x = ad.autodiff('x', 10)
    s = "str"
    with pytest.raises(ValueError):
        s**x
## Test power with a constant and an autodiff instance
def test_pow_result_adandconst():
    ad1 = ad.autodiff(name="x", val=2, der=1)
    ad2 = ad.autodiff(name="y", val=3, der=1)
    ad3 = ad1**(ad2**ad1)
    ad4 = (ad1**1.5)**ad2
    ad5 = (1.5**ad1)**ad2

    assert ad3.val == 512
    assert abs(ad3.der["x"] - 5812.9920480098718094) < 1E-10
    assert abs(ad3.der["y"] - 2129.3481386801519905) < 1E-10

    assert abs(ad4.val  - 22.627416997969520780) < 1E-10
    assert abs(ad4.der["x"] - 50.911688245431421756) < 1E-10
    assert abs(ad4.der["y"] - 23.526195443245132601) < 1E-10

    assert abs(ad5.val - 11.390625) < 1E-10
    assert abs(ad5.der["x"] - 13.855502991133679740) < 1E-10
    assert abs(ad5.der["y"] - 9.2370019940891198269) < 1E-10

def test_jacobian():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x*y
    assert f1.jacobian() == [['x', 'y'], [2, 10]]

def test_backprop():
    x = ad.autodiff('x', 3)
    f = admath.sin(admath.cos(admath.tan(admath.log(x))))
    # f.backprop() = {'x': array([-1.38686635])}
    assert f.backprop()['x'][0] == -1.3868663497018852
