import numpy as np
import matplotlib.pyplot as plt
from os.path import join, isdir
from os import mkdir
from tensorflow import reduce_mean, reshape, reduce_max
from tensorflow.keras import losses
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import json


class Postprocessing():
    def __init__(self, model, name, history, test_set, anomaly_set, data_type, test_shape, anomaly_shape, max, min):    # noqa: E501
        self.model = model
        self.name = name
        self.test_set = test_set
        self.anomaly_set = anomaly_set
        self.data_type = data_type
        self.test_shape = test_shape
        self.anomaly_shape = anomaly_shape
        self.history = history
        self.max = max
        self.min = min

    def plotting_graphs(self, path):
        # plot the training loss
        if self.history:
            fig, ax = plt.subplots()
            # print(self.history.history["loss"])
            plt.plot(self.history.history["loss"], label="Training Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Error")
            plt.grid('True')
            plt.title('Training Loss')
            plt.legend()
            plt.savefig(join(path, 'training_loss.png'))
            plt.close(fig)
            # saving the history loss for further comparision
            with open(join(path, 'history.json'), 'w') as file:
                json.dump(self.history.history["loss"], file, indent=4)

        # plot test set's loss distribution
        fig, ax = plt.subplots()
        test_reconstruction = self.model.predict(self.test_set)
        test_loss = losses.mean_squared_error(self.test_set, test_reconstruction)    # noqa: E501
        if self.name == 'vae':
            test_loss = self.vae_predict(test_loss, self.test_shape)
        else:
            test_loss = reshape(test_loss, (test_loss.get_shape()[0], -1))
            test_loss = reduce_mean(test_loss, axis=1)
        # print(test_loss)
        plt.hist(test_loss, bins=30)
        plt.xlabel("MSE")
        plt.ylabel("Number of samples")
        plt.title('Test set\'s loss distribution')
        plt.savefig(join(path, 'test_loss_distribution.png'))
        plt.close(fig)

        # anomaly set's loss distribution
        fig, ax = plt.subplots()
        anomaly_reconstruction = self.model.predict(self.anomaly_set)
        anomaly_loss = losses.mean_squared_error(self.anomaly_set, anomaly_reconstruction)    # noqa: E501
        if self.name == 'vae':
            anomaly_loss = self.vae_predict(anomaly_loss, self.anomaly_shape)
        else:
            anomaly_loss = reshape(anomaly_loss, (anomaly_loss.get_shape()[0], -1))    # noqa: E501
            anomaly_loss = reduce_mean(anomaly_loss, 1)
        plt.hist(anomaly_loss, alpha=0.5, label='anomaly', bins=30)
        plt.xlabel("MSE")
        plt.ylabel("Number of samples")
        plt.title('Anomaly set\'s loss distribution')
        plt.legend()
        plt.savefig(join(path, 'anomaly_loss_distribution.png'))
        plt.close(fig)
        return 0

    def calculate_metrics(self, path):
        labels = np.append(np.ones(self.anomaly_shape[0]).astype(bool), np.zeros(self.test_shape[0]).astype(bool))    # noqa: E501
        test = np.append(self.anomaly_set, self.test_set, axis=0)

        reconstructions = self.model(test)
        loss = losses.mean_squared_error(reconstructions, test)
        if self.name == 'vae':
            predictions = self.vae_predict(loss, labels.shape)
        else:
            predictions = reduce_mean(loss, axis=(1, 2))

        fpr, tpr, _ = roc_curve(labels, predictions)
        precision, recall, thresholds = precision_recall_curve(labels, predictions)

        # plot model roc curve
        fig, ax = plt.subplots()
        plt.plot(fpr, tpr, marker='.', label='ROC')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.title('ROC Curve')
        plt.legend()
        plt.grid()
        # show the plot
        plt.savefig(join(path, 'roc_curve.png'))
        plt.close(fig)

        # plot model precision-recall curve
        fig, ax = plt.subplots()
        plt.plot(recall, precision, marker='.', label='PR curve')
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Prescision')
        # show the legend
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid()
        # show the plot
        plt.savefig(join(path, 'pr_curve.png'))
        plt.close(fig)

        auc = roc_auc_score(labels, predictions)
        auc_1 = roc_auc_score(labels, predictions, max_fpr=0.1)
        return auc, auc_1, fpr, tpr, precision, recall, thresholds

    def save_results(self, path_to_save):
        path = path_to_save
        if not isdir(path):
            mkdir(path)

        print('Calculating and saving some graphs ...')
        self.plotting_graphs(path)
        print('Calculating and saving some metrics ...')
        auc, auc_1, fpr, tpr, precision, recall, thresholds = self.calculate_metrics(path)
        threshold = thresholds[np.argmax(precision+recall)]

        metrics = {
            'auc': auc,
            'auc_0.1': auc_1,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'threshold': threshold,
            'max': self.max,
            'min': self.min,
        }

        with open(join(path, 'metrics_detail.json'), 'w') as file:
            json.dump(metrics, file, indent=4)

        print('Done!!')
        return 0

    def vae_predict(self, loss, og_shape):
        num = loss.get_shape()[0]//og_shape[0]
        loss = reshape(loss, (-1, num))
        vae_loss = reduce_max(loss, axis=1)
        return vae_loss

    def calculate_bins(data):
        q25, q75 = np.percentile(data, [25, 75])
        bin_width = 2 * (q75 - q25) * len(data) ** (-1/3)
        bins = round((np.max(data) - np.min(data)) / bin_width)
        print("Freedman–Diaconis number of bins:", bins)
        return bins
