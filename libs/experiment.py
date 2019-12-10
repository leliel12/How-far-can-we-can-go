
import warnings

import numpy as np

import pandas as pd

from matplotlib import cm
from matplotlib import pyplot as plt

import seaborn as sns

import sklearn
from sklearn import feature_selection as fs
from sklearn import preprocessing as prp
from sklearn.model_selection import (
    KFold, StratifiedKFold, train_test_split)
from sklearn import metrics

from . import container


class Experiment(object):

    def __init__(self, data, clf, pcls, X_columns, y_column, clsnum,
                 ncls, sampler=None, verbose=True, real_y_column=None):
        self.data = data
        self.clf = clf
        self.pcls = pcls
        self.ncls = ncls
        self.sampler = sampler
        self.verbose = verbose
        self.X_columns = X_columns
        self.y_column = y_column
        self.clsnum = clsnum
        self.cfilter = [clsnum[pcls], clsnum[ncls]]

        if real_y_column is None:
            real_y_column = "{}_orig".format(y_column)
            columns = list(data.values())[0].columns
            if real_y_column not in columns:
                real_y_column = y_column
        self.real_y_column = real_y_column


    def experiment(self, x_train, y_train, x_test, y_test):
        if self.sampler:
            sampler = sklearn.clone(self.sampler)
            x_train, y_train = sampler.fit_sample(x_train, y_train)

        clf = sklearn.clone(self.clf)
        clf.fit(x_train, y_train)

        predictions = clf.predict(x_test)
        probabilities = clf.predict_proba(x_test)

        fpr, tpr, thresholds = metrics.roc_curve(
            y_test, 1.-probabilities[:,0], pos_label=self.cfilter[0])
        prec_rec_curve = metrics.precision_recall_curve(
            y_test, 1.- probabilities[:,0], pos_label=self.cfilter[0])
        roc_auc = metrics.auc(fpr, tpr)

        return container.Container({
                'fpr': fpr,
                'tpr': tpr,
                'thresh': thresholds,
                'roc_auc': roc_auc,
                'prec_rec_curve': prec_rec_curve,
                'y_test': y_test,
                'predictions': predictions,
                'probabilities': probabilities,
                'confusion_matrix': metrics.confusion_matrix(y_test, predictions)})

    def __call__(self, *args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.run(*args, **kwargs)


class WithAnotherExperiment(Experiment):

    def run(self, train_name):
        if isinstance(train_name, str):
            train_df = self.data[train_name]
            train_name = [train_name]
        else:
            train_df = pd.concat(
                [self.data[n] for n in train_name],
                ignore_index=True)

        results = []
        for test_name in sorted(self.data.keys()):

            if test_name not in train_name:
                 # retrieve the train data and test dataframe

                test_df = self.data[test_name]

                # filter only the important lines
                train_df = train_df[train_df.cls.isin(self.cfilter)]
                test_df = test_df[test_df.cls.isin(self.cfilter)]

                # split in np arrays
                x_train = train_df[self.X_columns].values
                y_train = train_df[self.y_column].values
                x_test = test_df[self.X_columns].values
                y_test = test_df[self.y_column].values
                y_test_real = test_df[self.real_y_column].values

                rst = self.experiment(x_train, y_train, x_test, y_test)
                rst.update({
                    'ids': test_df.id.values,
                    "test_size": len(test_df),
                    'test_name': test_name,
                    'y_test_real': y_test_real,
                    'train_name': " + ".join(train_name)})

                if self.verbose:
                    print "{} (TRAIN) Vs. {} (TEST)".format(rst.train_name, rst.test_name)
                    print metrics.classification_report(rst.y_test, rst.predictions)
                    print "-" * 80
                results.append(rst)
        return tuple(results)


class KFoldExperiment(Experiment):

    def run(self, subject, nfolds=10):
        # kfold
        skf = StratifiedKFold(n_splits=nfolds)

        subject_df = self.data[subject]
        subject_df = subject_df[subject_df.cls.isin(self.cfilter)]

        x = subject_df[self.X_columns].values
        y = subject_df[self.y_column].values
        ids = subject_df.id.values
        y_real = subject_df[self.real_y_column].values

        probabilities = None
        predictions = np.array([])
        y_testing = np.array([])
        y_testing_real = np.array([])
        u_ids = np.array([])

        for train, test in skf.split(x, y):
            x_train = x[train]
            y_train = y[train]
            x_test = x[test]
            y_test = y[test]
            y_test_real = y_real[test]

            rst = self.experiment(x_train, y_train, x_test, y_test)
            probabilities = (
                rst.probabilities if probabilities is None else
                np.vstack([probabilities, rst.probabilities]))
            predictions = np.hstack([predictions, rst.predictions])
            y_testing = np.hstack([y_testing, y_test])
            y_testing_real = np.hstack([y_testing_real, y_test_real])
            u_ids = np.hstack([u_ids, ids[test]])
            del rst

        fpr, tpr, thresholds = metrics.roc_curve(
            y_testing, 1.-probabilities[:,0],
            pos_label=self.cfilter[0])
        prec_rec_curve = metrics.precision_recall_curve(
            y_testing, 1.- probabilities[:,0],
            pos_label=self.cfilter[0])
        roc_auc = metrics.auc(fpr, tpr)

        if self.verbose:
            print metrics.classification_report(y_testing, predictions)
            print "-" * 80

        return container.Container({
            "test_name": "kfold",
            "test_size": len(subject_df),
            'ids': u_ids,
            'fpr': fpr,
            'tpr': tpr,
            'thresh': thresholds,
            'roc_auc': roc_auc,
            'prec_rec_curve': prec_rec_curve,
            'y_test': y_testing,
            'y_test_real': y_testing_real,
            'predictions': predictions,
            'probabilities': probabilities,
            'confusion_matrix': metrics.confusion_matrix(y_testing, predictions)})



def roc(results, cmap="plasma", save_to=None):
    cmap = cm.get_cmap(cmap)
    colors = iter(cmap(np.linspace(0, 1, len(results))))

    if isinstance(results, dict):
        for cname, res  in results.items():
            color = next(colors)
            label = '%s (area = %0.2f)' % (cname, res["roc_auc"])
            plt.plot(res["fpr"], res["tpr"], color=color, label=label)
    else:
        for res in results:
            cname = "Vs.{}".format(res.test_name)
            color = next(colors)
            label = '%s (area = %0.2f)' % (cname, res["roc_auc"])
            plt.plot(res["fpr"], res["tpr"], color=color, label=label)

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
    
    plt.show()
    

def roc(results, cmap="plasma", ax=None, save_to=None):
    if ax == None:
        ax = plt.gca()
    
    cmap = cm.get_cmap(cmap)
    colors = iter(cmap(np.linspace(0, 1, len(results))))

    if isinstance(results, dict):
        for cname, res  in results.items():
            color = next(colors)
            label = '%s (area = %0.2f)' % (cname, res["roc_auc"])
            ax.plot(res["fpr"], res["tpr"], color=color, label=label)
    else:
        for res in results:
            cname = "Vs.{}".format(res.test_name)
            color = next(colors)
            label = '%s (area = %0.2f)' % (cname, res["roc_auc"])
            ax.plot(res["fpr"], res["tpr"], color=color, label=label)

    ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    
    return ax