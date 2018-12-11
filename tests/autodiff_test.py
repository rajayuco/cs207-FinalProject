import pytest
import sys
import numpy as np
import pandas as pd

sys.path.append('..')
import numpy as np
from autodiffpy import autodiff as ad
from autodiffpy import autodiff_math as admath



## Test cases when autodiff value is vector, and derivatives are also vectors
def test_ad_der_vec():
    t = ad.autodiff('t',[1,2,3])
    assert all(t.der['t'] ==[1,1,1])

    z = ad.autodiff('z', [1,2,3], [1,1,1])
    assert all(z.der['z'] ==[1,1,1])

    e = ad.autodiff('e', [1,2,3], np.array([1,1,1]))
    assert all(e.der['e'] ==[1,1,1])

## Test equal to, not equal to
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

## Test non-equal variable names
def test_eq_varnames():
   x = ad.autodiff('x', [10, 20])
   y = ad.autodiff('y', [10, 20])
   z = ad.autodiff('z', [10, 20])
   f1 = x*y
   f2 = x*z
   assert f1 != f2
## Test equal to, not equal to
def test_eq_der():
   x = ad.autodiff('x', 2)
   y = ad.autodiff('y', 2)
   f1 = x+y
   f2 = x*y
   assert f1 != f2

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

## Test string format of autodiff instance
def test_str():
    t = ad.autodiff('t', 3)
    assert t.__str__() == f"value: {t.val}\nderivatives:{t.der}"


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

## Test true division with a constant or vector
def test_truediv_result_const():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x/y
    f2 = x/3
    assert f1.val == 5
    assert f1.der['x'] == 1/2
    assert f2.val == 10/3
    assert f2.der['x'] == 1/3

    t = ad.autodiff('t', [1,2,3])
    arr2 = [1,2,3]
    f1 = t/arr2
    f2 = arr2/x
    assert all(f1.val == [1,1,1])
    assert all(f1.der['t'] == [1, 0.5, 1/3])
    assert all(f2.val == [1/10, 2/10, 3/10])
    assert all(f2.der['x'] == [-1/100, -2/100, -3/100])

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

def test_mul_result_vec():
    x = ad.autodiff('x', [1,2,3])
    y = ad.autodiff('y', [3,4,5])
    f1 = x*y
    f2 = [1,2,3]*y
    dt = pd.DataFrame([[1,2,3],[4,5,6]])
    z = ad.autodiff('z', [1,2,3])
    f3 = z*dt

    assert all(f1.val == [3,8,15])
    assert all(f1.der['x'] == [3,4,5])
    assert all(f2.val == [3,8,15])
    assert all(f2.der['y'] == [1,2,3])
    assert all(f3.val == [14,32])
    assert all(f3.der['z'][0] == [1,2,3])
    assert all(f3.der['z'][1] == [4,5,6])

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

## Test jacobian() output
def test_jacobian():
    x = ad.autodiff('x', 10)
    y = ad.autodiff('y', 2)
    f1 = x*y
    assert f1.jacobian()["order"] == ['x', 'y']
    assert np.sum(f1.jacobian()["jacobian"] == np.array([2, 10])) == 2
    assert f1.jacobian(order=['y','x'])["order"] == ['y', 'x']
    assert np.sum(f1.jacobian(order=['y','x'])["jacobian"] == np.array([10, 2])) == 2

    a = ad.autodiff('a', [1, 2, 3])
    b = ad.autodiff('b', [-1, -2, -3])
    c = ad.autodiff('c', [-2, 5, 10])
    f2 = a*b - c
    assert f2.jacobian(order=['a', 'b'])["order"] == ['a', 'b']
    assert np.sum(f2.jacobian(order=['a', 'b'])["jacobian"] == np.array([[-1, -2, -3], [1, 2, 3]])) == 6

    with pytest.raises(KeyError):
        assert f2.jacobian(order=['a', 'd'])


def test_forwardprop():
    x = ad.autodiff('x', 3)
    y = ad.autodiff('y', 1)
    z = ad.autodiff('z', 2)
    e = ad.autodiff('e', 5)
    f1 = 4*x*y
    f2 = f1.forwardprop()
    assert f2 == f1

def test_forwardprop2():
    x = ad.autodiff('x', 3)
    y = ad.autodiff('y', 1)
    z = ad.autodiff('z', 2)
    e = ad.autodiff('e', 5)
    f1 = 4/x
    f2 = f1.forwardprop()

    f3 = 2**y
    f4 = f3.forwardprop()

    f5 = y - z
    f6 = f5.forwardprop()

    f7 = z**x
    f8 = f7.forwardprop()

    f9 = -x
    f10 = f9.forwardprop()

    assert f2 == f1
    assert f3 == f4
    assert f5 == f6
    assert f7 == f8
    assert f9 == f10

## Test backprop() output with hyperbolic functions
def test_backprop_hyperbolic():
    x = ad.autodiff('x', 1)
    y = ad.autodiff('y', 2)
    z = ad.autodiff('z', 3)
    f1 = admath.sinh(x)*admath.cosh(y)*admath.tanh(z)
    print(f1.backprop(y_true=2))
    assert pytest.approx(f1.backprop(y_true=2)[0]['x'][0]) == f1.back_der*np.tanh(z.val)*np.cosh(y.val)*np.cosh(x.val)
    assert pytest.approx(f1.backprop(y_true=2)[0]['y'][0]) == f1.back_der*np.tanh(z.val)*np.sinh(x.val)*np.sinh(y.val)


def test_backprop_sincostanlog():
    x = ad.autodiff('x', 3)
    f = admath.sin(admath.cos(admath.tan(admath.log(x))))
    # f.backprop() = {'x': array([-1.38686635])}
    assert pytest.approx(f.backprop(y_true=[2])[0]['x'][0]) == -1.386866349701885*f.back_der

## Test gradient_descent() with MSE loss
def test_gradient_descent_MSE():
   x = np.array([[1,-2,1],[3,0,4]]) #Data
   w = ad.autodiff('w', [3, -1, 0]) #Weights

   # Set up parameters for gradient descent
   max_iter = 5000
   beta = 0.005
   f = w*x
   y_act = [5.5,9.5]
   tol = 1E-8
   loss = "MSE"

   # Run gradient descent
   g = ad.gradient_descent(f, y_act, beta=beta, loss=loss, max_iter=max_iter, tol=tol)

   # Assert correct values within tolerance
   assert ((g['f'].val[0] - y_act[0])**2 + (g['f'].val[1] - y_act[1])**2)/len(y_act) <= tol
   #assert pytest.approx(np.abs(g['f'].val[1] - y_act[1]) <= np.ones(len(y_act))*tol)


## Test gradient_descent() with MAE loss
def test_gradient_descent_MAE():
    x = np.array([[5,-2],[3,-4]]) #Data
    w = ad.autodiff('w', [3, 0.5]) #Weights

    # Set up parameters for gradient descent
    max_iter = 10000
    beta = 0.1
    f = 1 + ad.exp(-1*w*x)
    y_act = [1.0, 1.05]
    tol = 1E-4
    loss = "MAE"

    # Run gradient descent
    g = ad.gradient_descent(f, y_act, beta=beta, loss=loss, max_iter=max_iter, tol=tol)

   # Assert correct values within tolerance
    assert (np.absolute(g['f'].val[0] - y_act[0]) + np.absolute(g['f'].val[1]-y_act[1]))/len(y_act) <= tol
    #assert g["loss_array"][-1] <= tol


## Test gradient_descent() with RMSE loss
def test_gradient_descent_RMSE():
    x = np.array([[2,0],[5,1]]) #Data
    w = ad.autodiff('w', [0.6,0.4]) #Weights

    # Set up parameters for gradient descent
    max_iter = 40000
    beta = 0.00001
    f = 3 + w*x/2.0
    y_act = [3,4]
    tol = 0.2
    loss = "RMSE"

   # Run gradient descent
    g = ad.gradient_descent(f, y_act, beta=beta, loss=loss, max_iter=max_iter, tol=tol)

   # Assert correct values within tolerance
    assert np.sqrt(((g['f'].val[0] - y_act[0])**2 + (g['f'].val[1] - y_act[1])**2)/len(y_act)) <= tol

## Test gradient_descent() with RMSE loss
def test_gradient_descent_weightname():
    x = np.array([[2,0],[5,1]]) #Data
    w = ad.autodiff('t', [0.6,0.4]) #Weights

    # Set up parameters for gradient descent
    max_iter = 40000
    beta = 0.00001
    f = 3 + w*x/2.0
    y_act = [3,4]
    tol = 0.2
    loss = "RMSE"
    with pytest.raises(ValueError):
    # Run gradient descent
        g = ad.gradient_descent(f, y_act, beta=beta, loss=loss, max_iter=max_iter, tol=tol)
