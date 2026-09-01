## Results

The performance of the four anomaly detection approaches was evaluated on
numerical datasets constructed from standard-cell timing and power data
extracted from `.lib` file matrices.

The experiments considered datasets with different numbers of features and
different anomaly rates. Two main aspects were evaluated: model training time
and anomaly detection performance using the ROC-AUC metric.

### Training Time

The table below summarizes the training time of the evaluated models for
different dataset configurations.

| Dataset Size | Features | Anomaly Rate | K-NN (s) | LOF (s) | Isolation Forest (s) | Random Forest (s) |
|--------------:|---------:|-------------:|---------:|--------:|----------------------:|------------------:|
| 12,720 | 6  | 3%  | 0.08 | 0.42 | 0.55 | 2.69 |
| 12,720 | 6  | 10% | 0.09 | 0.44 | 0.64 | 2.54 |
| 16,900 | 9  | 10% | 0.15 | 0.79 | 0.68 | 5.07 |
| 16,900 | 9  | 25% | 0.16 | 1.25 | 0.62 | 5.27 |
| 12,720 | 12 | 25% | 0.15 | 0.88 | 0.84 | 3.79 |
| 12,720 | 12 | 40% | 0.15 | 0.92 | 0.58 | 3.77 |

The results show that **K-NN required the shortest training time** across the
tested configurations, while **Random Forest generally required the longest
training time**. LOF and Isolation Forest showed intermediate training times.

### ROC-AUC Performance

The anomaly detection performance was evaluated using the
**Receiver Operating Characteristic – Area Under the Curve (ROC-AUC)** metric.

The experiments were conducted using datasets with different feature-space
dimensionalities and anomaly rates. The ROC-AUC results demonstrate that all
four approaches were able to distinguish anomalous observations from normal
observations with varying levels of effectiveness.

Overall, the obtained ROC-AUC values were approximately in the **0.70–0.83**
range. The best-performing configurations achieved ROC-AUC values of about
**0.83**, indicating good discrimination between normal and anomalous
observations.

The results also show that model performance depends on both the
**dimensionality of the feature space** and the **proportion of anomalies** in
the dataset. Therefore, no single method consistently dominates across every
experimental configuration.

### Summary

The experiments demonstrate that:

- **K-NN** provides the lowest training time among the evaluated methods.
- **Random Forest** has the highest computational cost in terms of training
  time in the tested configurations.
- **LOF** and **Isolation Forest** provide intermediate training times.
- The evaluated methods achieve ROC-AUC values indicating meaningful anomaly
  detection capability.
- The ROC-AUC performance varies depending on the number of features and the
  anomaly rate.
- The experiments illustrate the trade-off between **computational efficiency**
  and **anomaly detection performance**.
