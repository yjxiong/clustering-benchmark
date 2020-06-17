import argparse
import json
import numpy as np
from scipy.cluster.vq import vq, kmeans
from clustering_benchmark.io import read_groundtruth_file_v1

from clustering_benchmark import ClusteringBenchmark


parser = argparse.ArgumentParser('Run the clustering benchmark')
parser.add_argument('--data-file', type=str, required=True,
                    help='the file storing both features and groundtruth')

if __name__ == '__main__':

    args = parser.parse_args()

    # run a simple kmeans clustering with 9 clusters
    raw_data = json.load(open(args.data_file))

    feature_pairs = [(k, v['embedding']) for k, v in raw_data.items()]
    features = np.array([x[1] for x in feature_pairs])

    codebook, distortion = kmeans(features, 9)
    predicted_labels, _ = vq(features, codebook)

    prediction_dict = {
        item[0]: label for item, label in zip(feature_pairs, predicted_labels)
    }

    # run the benchmark
    gt_data = read_groundtruth_file_v1(args.data_file)
    bm = ClusteringBenchmark(gt_data)

    scores = bm.evaluate_vmeasure(prediction_dict)

    print("Evaluation results:")
    print(scores)








