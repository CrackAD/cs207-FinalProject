## Software Organization

Our directory structure will look like:

```
EasyDiff/
		easydiff/
				__init__.py
				ad.py
				var.py
				tests/
					__init__.py
					test.py
		README.md
		setup.py
		LICENSE
```
In the directory, we have two python modules `ad.py` and `var.py`. 

* `ad.py`: all automatic differentiation calculations
* `var.py`: dual number constructions and basic math operations overloaded

We also plan on including dependencies `numpy` and `math` to overload elementory operations.

Our test suite will be in the `test` folder, and we will implement `pytest` to write comprehensive tests to provide full coverage for our code. We will also use `TravisCI` and `CodeCov` to automate the testing process.

PyPI will be used to distribute our package, as it enables the user to install our package using `pip`.