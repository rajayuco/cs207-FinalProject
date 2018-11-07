
import sys
import os

sys.path.append('..')

import pytest
from autodiffpy import autodiff as ad



## Test true division with an autodiff instsance
def test_truediv_result_ad():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x/y
    assert f1.val == 5
    assert f1.der['x'] == 1/2
    assert f1.der['y'] == 10/4

## Test true division with a constant
def test_truediv_result_const():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f2 = x/3
    assert f2.val == 10/3
    assert f2.der['x'] == 1/3

## Test reverse division with an autodiff instance
def test_rtruediv_result_ad():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = y/x
    assert f1.val == 2/10
    assert f1.der['x'] == 2/10**2
    assert f1.der['y'] == 1/10

## Test multiplication with an autodiff instance
def test_mul_result_ad():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x*y
    assert f1.val == 20
    assert f1.der['x'] == 2
    assert f1.der['y'] == 10

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
    assert -x.val == -10
    assert -x.der['x'] == -1

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
