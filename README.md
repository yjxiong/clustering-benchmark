clustering-benchmark
----
toolkit for benchmark clustering algorithms

### Install 
Run 
```bash
python setup.py install
```

### Usage
After installed, use it as a library

```python
from clustering_benchmark import ClusteringBenchmark

bm = ClusteringBenchmark(gt_file=gt_filename)

scores = bm.evaluate_vmeasure(prediction)

print("Evaluation results:")
print(scores)
```

For details, please see the script in `tools/run_clustering_test.py`

