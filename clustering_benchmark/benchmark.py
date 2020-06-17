
from sklearn import metrics


class ClusteringBenchmark:
    """
    The benchmark for clustering

    At initialization, it requires the groundtruth data

    To evaluate a metric, it takes as input the prediction data.

    Both of them should be Python dicts with the following format

    {
        item_id: item_label,
    }

    where item_label is a integer showing the cluster assignment or the groundtruth class label
    """

    def __init__(self, gt_data):
        self._gt = gt_data

    def evaluate_vmeasure(self, predictions):
        results = self._basic_stats(predictions)

        key_list = list(predictions.keys())

        gt_labels = [self._gt[k] for k in key_list]
        pred_labels = [predictions[k] for k in key_list]

        h = metrics.homogeneity_score(gt_labels, pred_labels)
        c = metrics.completeness_score(gt_labels, pred_labels)

        v = 2 / (1 / c + 1 / h)

        results.update(
            {
                'h-score': h,
                'c-score': c,
                'v-meansure': v
            }
        )

        return results

    def _basic_stats(self, predictions):
        num_gt = len(set(v for v in self._gt.values()))
        num_pred = len(set(v for v in predictions.values()))

        return {'#gt clusters': num_gt, '#pred clusters': num_pred}
