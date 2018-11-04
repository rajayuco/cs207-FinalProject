
###Below imports necessary functions
from autodiff_jnp import autodiff as adiff

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



