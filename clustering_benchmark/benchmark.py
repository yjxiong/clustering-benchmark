
import numpy as np
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix


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

        pred_labels, gt_labels = self._gen_list(predictions)

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

    def evaluate_fowlkes_mallows_score(self, predictions):
        """
        Calculate FMI clustering metrics.
        Adopted from https://github.com/sneakerkg/learn-to-cluster/blob/vegcn_dgl/evaluation/metrics.py#L41
        :param predictions:
        :return:
        """
        results = self._basic_stats(predictions)

        pred_labels, gt_labels = self._gen_list(predictions)

        n_samples = len(gt_labels)
        c = contingency_matrix(gt_labels, pred_labels, sparse=True)
        tk = np.dot(c.data, c.data) - n_samples
        pk = np.sum(np.asarray(c.sum(axis=0)).ravel() ** 2) - n_samples
        qk = np.sum(np.asarray(c.sum(axis=1)).ravel() ** 2) - n_samples

        avg_pre = tk / pk
        avg_rec = tk / qk
        fscore = 2. * avg_pre * avg_rec / (avg_pre + avg_rec)

        results.update({
            'fmi_score': fscore,
            'pair_precision': avg_pre,
            'pair_recall': avg_rec
        })

        return results

    def _basic_stats(self, predictions):
        num_gt = len(set(v for v in self._gt.values()))
        num_pred = len(set(v for v in predictions.values()))

        return {'#gt clusters': num_gt, '#pred clusters': num_pred}

    def _gen_list(self, predictions):
        key_list = list(predictions.keys())

        gt_labels = [self._gt[k] for k in key_list]
        pred_labels = [predictions[k] for k in key_list]

        return pred_labels, gt_labels

