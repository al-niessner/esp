### Adding an estimator

Estimators are of two types: 1) Stellar estimators producing a value
per star and 2) Planetary estimators producing a value per planet

They can be implemented by either subclassing the corresponding estimator
type, or if, subclassing is an unfamiliar concept, can be implemented
by defining a standard Python function that accepts the correct arguments.

**Note:** Estimators can rely upon previous estimators but be aware that
estimators are calculated in sequential order and each estimator only
has access to the previously defined estimator values. This prevents uncomputable
estimators that mutually rely upon each other.


#### Instructions using simple function

1. In `estimators.py`, define a new function that accepts `priors` & `ests` parameters
for stellar estimators and `priors`, `ests` & `planet` parameters for planetary
estimators. Note: Define the `ests` parameter as `_ests` if you will not use previous estimates to prevent the pylint unused-variable warning. Alternatively, using the Python `class` estimator approach below will more cleanly prevent this warning.
2. If your estimator function was defined outside of `estimators.py` then in `core.py`, import your new estimator. Otherwise, no action is needed and you can reference the function by `ancestor.your_function_name` in step 3 (where it says `my_imported_method`).
3. In `core.py`, include the estimator in the `getestimators()` function by
adding a line of the form `PlEstimator(name='example', 'descr='Example estimator', units='Ex', method=my_imported_method)` for planetary estimators and something of
an identical form except changing `PlEstimator` to `StEstimator` for
stellar estimators.


#### Instructions using Python `class`

1. In `estimators.py`, define a new estimator by subclassing from
`StEstimator` for stellar estimators or `PlEstimator` for planetary
estimators. The `run()` method should produce the estimate to store
in the state vector.
2. In `core.py`, import your new estimator or reference by `ancestor.YourEstimator`
and then include it in
the `getestimators()` function in the appropriate list (i.e. stellar
or planetary).

Planetary estimator template:
```
class MinimalPlEstimator(PlEstimator):
    def __init__(self):
        PlEstimator.__init__(self, name='minimal', descr='Minimal estimator')

    def run(self, priors, ests, pl):
        return 1.0
```

Stellar estimator template:
```
class MinimalStEstimator(StEstimator):
    def __init__(self):
        StEstimator.__init__(self, name='minimal', descr='Minimal estimator')

    def run(self, priors, ests):
        return 1.0
```

### Estimator Output Format

**Failing Output** - If an estimator cannot generate an estimate for a given target, it needs to return `None` so the system will correctly prevent it from being stored.

**Basic Output** - Estimators returning a single value such as a number or a string can simply return this value from the function and it will be stored at the SV key.

**Confidence Interval** - Estimators with a confidence interval on their estimates can return a Python dictionary object of the form `{'val': estimate, 'uperr': uperr, 'lowerr': lowerr}`.

### Estimator Parameter Descriptions
For both `StEstimator` and `PlEstimator` types, they support the following
parameters when they are initialized.
* `name` - The name of your estimator. Will be used also as the key in the SV.
* `descr` - A short description providing the context of what your estimator computes.
* `method` - A function accepting the correct parameters as described in above sections which performs the actual parameter estimation. 
* (opt) `plot` - Currently either `'hist'` or `'bar'` depending upon if the population distribution of this estimate should be displayed by a histogram or a bar graph.
* (opt) `units` - The units of your estimator as a string.
* (opt) `scale` - Used in conjunction with `plot='bar'` and specifies the order of the different elements in the bar graph (e.g. `['A', 'B', 'C']`). 
* (opt) `ref` - The name of the reference the estimator formulation was described in.

