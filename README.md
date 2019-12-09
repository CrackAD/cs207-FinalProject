# EasyDiff [![Build Status](https://travis-ci.com/CrackAD/cs207-FinalProject.svg?branch=master)](https://travis-ci.com/CrackAD/cs207-FinalProject) [![codecov](https://codecov.io/gh/CrackAD/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/CrackAD/cs207-FinalProject)


EasyDiff is an automatic differentiation python library with forward and reverse mode supported. EasyDiff is developed as a Harvard CS207 (19Fall) course project by **Group 18: [Yang Zhou](https://github.com/YangZhou1997), [Ruby Zhang](https://github.com/Ruby122), [Kangli Wu](https://github.com/KangliMalorie), and [Emily Gould](https://github.com/coolcilantro).** Check our [documentation](./docs/documentation.md) for more details.  

## How to use CrackAD

#### Installation

##### Option One: Downloading Using Pip

**Get The Package**

Simply open your terminal and type the following command:
```
pip install easydiff
```
**Update The Package**

To get new releases, paste this into your terminal:
```
pip install easydiff --upgrade
```
We highly recommend installing the package with `pip`. Yet, if that doesn't work for you, you can still get our package with the second option below.

##### Option Two: Downloading From GitHub

**Clone the Repository**

Clone our GitHub repository and navigate into this directory in your terminal:
```
git clone https://github.com/CrackAD/cs207-FinalProject.git
```
In order to use the CrackAD package, you'll need to create a virtual environment. We recommend conda because it is both a package and environment manager and is language agnostic. Please run the following commands in a terminal:

**Create Conda Environment** 

Create an environment with the command, where `env_name` is the name of your choice. Since our package requires the  `NumPy` package, we also install it at this step: 
```
conda create --name env_name python numpy
```

**Activate the Environment**

To activate the Conda environment just created, run the following line:
```
source activate env_name
```
Or 
```
conda activate env_name
```
Yet, it is possible that the second one doesn't work because conda will complain that the shell hasn't been configured to use conda activate. So we would recommend using the first line.

**Install Packages**

If you haven't installed `NumPy` in the first step, or if you ever need to install another package, simply do the following:
```
conda install numpy
```

To check whether the installation succeeded, we could list out all installed packages in this environment:
```
conda list
```

If the `conda install` did not work, try `pip install`:
```
pip install Numpy
```
Note that it is suggested to always try `conda install` first.

#### Demonstration

To use CrackAD, create a .py file (eg, `driver.py`) with the following lines of code:
```
from EasyDiff.ad import AD
from EasyDiff.var import Var
from EasyDiff.rev_var import Rev_Var
from EasyDiff.ad import AD_Mode
import numpy as np

# test forward mode. 
# give it a function of your choice
func = lambda x,y: Var.log(x) ** Var.sin(y)

# give the initial values to take the derivatives at
ad = AD(vals=np.array([2, 2]), ders=np.array([1, 1]), mode=AD_Mode.FORWARD)

# calculate and print the derivatives
print("Var.log(x) ** Var.sin(y): {}".format(vars(ad.auto_diff(func))))

# test reverse mode. 
func = lambda x,y: Rev_Var.log(x) ** Rev_Var.sin(y)
ad = AD(vals=np.array([2, 2]), ders=np.array([1, 1]), mode=AD_Mode.REVERSE)
print("Rev_Var.log(x) ** Rev_Var.sin(y): {}".format(vars(ad.auto_diff(func))))
```
Then, you can run the file in a terminal as follows:
```
python3 driver.py
```
