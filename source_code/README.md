## Source Code for Reproducing DeepNSD Source Data

### 'Pressures' Quick Start

In [*pressures*](./pressures/), you will find the main source code for the reproducing the results in "What can 1.8 billion regressions tell us about the pressures shaping high-level visual representation in brains and machines?" by Colin Conwell, Jacob S. Prince, Kendrick N. Kay, George A. Alvarez, and Talia Konkle (Nature Communications, In Press).

Assuming your current working directory is this directory, you can get (main) results for a target model by running the following.

```python
from pressures.main_analysis import *
model_uid = 'alexnet_classification'
benchmark = 'shared1000_OTC-only'

results = run_model_on_benchmark(model_uid, benchmark)
```

More details on all aspects of our analysis are available in individual .py files (e.g. in [aux_scripts]['./pressures/aux_scripts]) or jupyter [notebooks](./pressures/notebooks/).


### Work in Progress: Versioning

We hope to soon have either a Dockerfile or Conda env file that will install the versioned packages we used when running this analysis. Stay tuned!