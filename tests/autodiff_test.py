import pytest
import autodiffpy as ad

## example cases
x = ad.autodiff('x', 10)
y = ad.autodiff('y', 2)


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

#Test addition
def test_autodiff_add1():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=20.5, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad3 = ad1 + ad2 + ad1
	
	#Check results
	checkres = []
	checkres.append(ad3.val == 44)
	checkres.append(ad3.der["x"] == 2)
	checkres.append(ad3.der["y"] == 1)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))


#Test addition
def test_autodiff_add2():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=20.5, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad4 = ad1 + 5.5 + ad2
	
	#Check results
	checkres = []
	checkres.append(ad4.val == 29)
	checkres.append(ad4.der["x"] == 1)
	checkres.append(ad4.der["y"] == 1)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))


#Test addition
def test_autodiff_add3():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=20.5, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad5 = 5.5 + ad1 + ad2
	
	#Check results
	checkres = []
	checkres.append(ad5.val == 29)
	checkres.append(ad5.der["x"] == 1)
	checkres.append(ad5.der["y"] == 1)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))


#Test subtraction
def test_autodiff_sub1():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=2.5, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad3 = ad1 - ad2 - ad1
	
	#Check results
	checkres = []
	checkres.append(ad3.val == -3)
	checkres.append(ad3.der["x"] == 0)
	checkres.append(ad3.der["y"] == -1)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))


#Test subtraction
def test_autodiff_sub2():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=2.5, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad4 = ad1 - 5.5 - ad2
	
	#Check results
	checkres = []
	checkres.append(ad4.val == -6)
	checkres.append(ad4.der["x"] == 1)
	checkres.append(ad4.der["y"] == -1)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))


#Test subtraction
def test_autodiff_sub3():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=2.5, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad5 = 5.5 - ad1 - ad2
	
	#Check results
	checkres = []
	checkres.append(ad5.val == 0)
	checkres.append(ad5.der["x"] == -1)
	checkres.append(ad5.der["y"] == -1)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))


#Test power
def test_autodiff_pow1():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=2, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad3 = (ad1**ad2)**ad1
	
	#Check results
	checkres = []
	checkres.append(ad3.val == 64)
	checkres.append(abs(ad3.der["x"] - 5812.9920480098718094) < 1E-16)
	checkres.append(abs(ad3.der["y"] - 2129.3481386801519905) < 1E-16)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))


#Test power
def test_autodiff_pow2():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=2, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad4 = (ad1**1.5)**ad2
	
	#Check results
	checkres = []
	checkres.append(abs(ad4.val  - 10.374716437208077327) < 1E-16)
	checkres.append(abs(ad4.der["x"] - 17.507333987788630490) < 1E-16)
	checkres.append(abs(ad4.der["y"] - 9.8407672680019981255) < 1E-16)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))


#Test power
def test_autodiff_pow3():
	#Generate autodiff instances
	ad1 = adiff(name="x", val=2, der=1)
	ad2 = adiff(name="y", val=3, der=1)
	
	#Add them together
	ad5 = (1.5**ad1)**ad2
	
	#Check results
	checkres = []
	checkres.append(abs(ad5.val - 25.62890625) < 1E-16)
	checkres.append(abs(ad5.der["x"] - 124.69952692020311766) < 1E-16)
	checkres.append(abs(ad5.der["y"] - 57.623417001265194147) < 1E-16)
	
	#Assert results are all correct
	assert (len(checkres) == sum(checkres))



