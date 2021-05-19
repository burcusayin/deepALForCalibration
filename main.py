# !pip install transformers
# !pip install sklearn
# !pip install netcal

import os
import random
import numpy as np
import pandas as pd
from random import shuffle

from netcal.metrics import ECE
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import brier_score_loss
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer

# setup random seed
seed = 2020
np.random.seed(seed)
random.seed(seed)

# define the paths to your data
# data_folder = 'drive/My Drive/Colab Notebooks/deepALForCalibration/datasets/binary/economic_news/'
# specify the path to the folder where you keep your datasets
# dataToTrain = '4_train_indexed_economic_news_binary.csv'            # file name for your training data
# dataToVal = '4_val_indexed_economic_news_binary.csv'                 # file name for your validation data
# dataToTest = '4_test_indexed_economic_news_binary.csv'                # file name for your test data
data_folder = '/Users/fabio.casati/Documents/dev/labdev/data/az/'  # specify the path to the folder where you keep your datasets
dataToTrain = 'df0L.csv'  # file name for your training data
dataToVal = 'df0L.csv'  # file name for your validation data
dataToTest = 'df0L.csv'  # file name for your test data

res_path = './'  # specify the path to keep results
logfile_name = "azlog.csv"  # specify the name of the result file

# columns of the csv file used in the experiments: text/content for each item, gold labels for each item, confidence scores for each class, ID of each item
# specify the column names of your data
iID = 'sys_id'  # give each item an ID, it will be used during active learning
goldLabel = 'assignment_group'  # define the name of column where you keep the gold labels of your data
txt = 'short_description'  # define the name of column where you keep the items
testGoldLabel = 'assignment_group'


features_and_labels = set([iID,goldLabel,txt, testGoldLabel])
# specify the active learning strategy you want to use
al_strategy = 'diversity'
# resDiversity = 'drive/My Drive/Colab Notebooks/deepALForCalibration/res/diversityRankings_MLP_3x100_D3_resampledByOneSideSelection.csv'
# al_strategy = 'uncertainty'
# al_strategy = 'random'

logfile_name = al_strategy + logfile_name

# PARAMETERS
# num_labels = 2  # number of classes in your data
#
# mClass = [0, 1]  # define all of possible classes

df = pd.read_csv(data_folder + dataToTrain)
df = df[features_and_labels].fillna('_NAN')

mClass = df[goldLabel].unique()
num_labels = len(mClass)


minimum_training_items = 66  # minimum number of training items before we first train a model
alBatchNum = 10  # define the total number of batches in active learning pipeline
alBatchSize = 180  # define the size of one batch in active learning pipeline
maxTfIdfFeat = 1024  # define the maximum number of features for tfidf

# MLP for Dataset 1
# model = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=500, alpha=0.001, activation = 'tanh', solver='sgd')
# MLP for Dataset 2
# model = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=500, alpha=0.05, activation = 'tanh', solver='sgd')
# MLp for Dataset 3
# model = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), max_iter=500, alpha=0.05, activation = 'relu', solver='adam')  # define the classification model you want to use
# MLP for Dataset 4
# model = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=500, alpha=0.001, activation = 'tanh', solver='sgd')
# MLP for Dataset 8
model = MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=500, alpha=0.05, activation='relu',
                      solver='adam')

poolDataEmb_train = np.array([])
poolDataEmb_val = np.array([])
poolDataEmb_test = np.array([])

# create log file
res_path += logfile_name
if len(mClass) == 2:
    with open(res_path, 'w') as f:
        c = 'alBatch, sampledIndices, pre_train, rec_train, f01_train, f1_train, f10_train, ece_train, brier_train, pre_val, rec_val, f01_val, f1_val, f10_val, ece_val, brier_val, pre_test, rec_test, f01_test, f1_test, f10_test, ece_test, brier_test'
        f.write(c + '\n')
else:
    with open(res_path, 'w') as f:
        c = 'alBatch, sampledIndices, pre_train, rec_train, f01_train, f1_train, f10_train, ece_train, pre_val, rec_val, f01_val, f1_val, f10_val, ece_val, pre_test, rec_test, f01_test, f1_test, f10_test, ece_test'
        f.write(c + '\n')

# specify data directories
unlabeled_data_dir = data_folder + dataToTrain
validation_data_dir = data_folder + dataToVal
test_data_dir = data_folder + dataToTest


class DiversitySampling():

    def __init__(self, verbose):
        self.verbose = verbose

    def get_validation_rankings(self, model, validation_data, val_emb):
        """Get model outliers from unlabeled data

        Keyword arguments:
            model -- current Machine Learning model for this task
            unlabeled_data -- data that does not yet have a label
            validation_data -- held out data drawn from the same distribution as the training data
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)

        An outlier is defined as
        unlabeled_data with the lowest average from rank order of logits
        where rank order is defined by validation data inference

        """

        validation_rankings = []  # 2D array, every neuron by ordered list of output on validation data per neuron

        # Get per-neuron scores from validation data
        if self.verbose:
            print("Getting neuron activation scores from validation data")

        pred = model.predict_proba(val_emb)

        v = 0
        for neuron_outputs in pred:
            # initialize array if we haven't yet
            if len(validation_rankings) == 0:
                for output in list(neuron_outputs):
                    validation_rankings.append([0.0] * len(validation_data))

            n = 0
            for output in list(neuron_outputs):
                validation_rankings[n][v] = output
                n += 1
            v += 1

        # Rank-order the validation scores
        v = 0
        for validation in validation_rankings:
            validation.sort()
            validation_rankings[v] = validation
            v += 1

        return validation_rankings

    def get_rank(self, value, rankings):
        """ get the rank of the value in an ordered array as a percentage

        Keyword arguments:
            value -- the value for which we want to return the ranked value
            rankings -- the ordered array in which to determine the value's ranking

        returns linear distance between the indexes where value occurs, in the
        case that there is not an exact match with the ranked values
        """

        index = 0  # default: ranking = 0

        for ranked_number in rankings:
            if value < ranked_number:
                break  # NB: this O(N) loop could be optimized to O(log(N))
            index += 1

        if (index >= len(rankings)):
            index = len(rankings)  # maximum: ranking = 1

        elif (index > 0):
            # get linear interpolation between the two closest indexes

            diff = rankings[index] - rankings[index - 1]
            perc = value - rankings[index - 1]
            linear = perc / diff
            index = float(index - 1) + linear

        absolute_ranking = index / len(rankings)

        return (absolute_ranking)

    def get_model_outliers(self, dataPool, model, unlabeled_data, unl_emb, validation_data, val_emb, number):
        """Get model outliers from unlabeled data

        Keyword arguments:
            model -- current Machine Learning model for this task
            unlabeled_data -- data that does not yet have a label
            validation_data -- held out data drawn from the same distribution as the training data
            number -- number of items to sample
            limit -- sample from only this many items for faster sampling (-1 = no limit)

        An outlier is defined as
        unlabeled_data with the lowest average from rank order of logits
        where rank order is defined by validation data inference

        """

        # Get per-neuron scores from validation data
        validation_rankings = self.get_validation_rankings(model, validation_data, val_emb)

        # Iterate over unlabeled items
        if self.verbose:
            print("Getting rankings for unlabeled data")

        outliers = []
        pred = model.predict_proba(unl_emb)

        itID = 0
        for neuron_outputs in pred:
            n = 0
            ranks = []
            for output in neuron_outputs:
                rank = self.get_rank(output, validation_rankings[n])
                ranks.append(rank)
                n += 1
            avgRank = 1 - (sum(ranks) / len(neuron_outputs))  # average rank
            currentRow = unlabeled_data.iloc[[itID]].reset_index(drop=True)
            rowIndex = currentRow.itemID.item()
            row = dataPool.loc[dataPool[iID] == rowIndex]
            row['avgRank'] = avgRank
            outliers.append(row.values.flatten().tolist())
            itID += 1
        outliers.sort(reverse=True, key=lambda x: x[-1])
        return outliers[:number:]


def random_sampling(unknownIndices, nQuery):
    '''Randomly samples the points'''
    query_idx = random.sample(range(len(unknownIndices)), nQuery)
    selectedIndex = unknownIndices[query_idx]
    return selectedIndex


# uncertainty sampling yaz
def uncertainty_sampling(dataPool, model, unl_emb, number):
    '''Points are sampled according to uncertainty sampling criterion'''

    pred = model.predict_proba(unl_emb)
    uncertainty_scores = 1 - pred.max(axis=1)
    score_indices = np.argsort(uncertainty_scores)
    return score_indices[-number:]


## Feature Preparation
def prepare_features(X_train, min_df=2, max_features=None, ngram_range=(1, 3)):
    # compute tfidf features
    tfidf = TfidfVectorizer(min_df=min_df, max_features=max_features,
                            strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                            ngram_range=ngram_range, use_idf=1, smooth_idf=1, sublinear_tf=1,
                            stop_words=None, lowercase=False)

    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    return X_train_tfidf


class Data():

    def __init__(self, filename):

        # each dataset will have a pool of data, together with their IDs and gold labels
        self.poolData = np.array([])
        self.poolGoldLabels = np.array([])

        dt = pd.read_csv(filename)
        dt = dt[features_and_labels].fillna('')
        indices = dt[iID].values
        y = dt[goldLabel].values
        X = prepare_features(dt[txt].tolist(), min_df=0, max_features=maxTfIdfFeat, ngram_range=(1, 3))
        self.data = dt
        self.poolDataEmb = X
        self.poolGoldLabels = y
        self.poolDataIndices = indices

    def setStartState(self, nStart):
        ''' This functions initialises fields indicesKnown and indicesUnknown which contain the datapoints having final labels(known) and still explorable(unknown) ones.
        Input:
        nStart -- number of labelled datapoints (size of indicesKnown)
        '''
        self.nStart = nStart
        self.indicesKnown = np.array([])
        self.indicesUnknown = np.array([])

        # get predefined points so that all classes are represented and initial classifier could be trained.

        for cls in mClass:
            indices = np.array(np.where(self.poolGoldLabels == cls)).tolist()[0]
            sampledIndices = random.sample(indices, nStart // len(mClass))
            dataIndices = np.array(self.poolDataIndices)
            if self.indicesKnown.size == 0:
                self.indicesKnown = dataIndices[sampledIndices]
            else:
                self.indicesKnown = np.concatenate(([self.indicesKnown, dataIndices[sampledIndices]]));
        for i in self.poolDataIndices:
            if i not in self.indicesKnown:
                if self.indicesUnknown.size == 0:
                    self.indicesUnknown = np.array([i])
                else:
                    self.indicesUnknown = np.concatenate(([self.indicesUnknown, np.array([i])]));


# function to calculate the ECE score
def ece_score(y_true, y_prob, n_bins=10):
    ece = ECE(n_bins)
    ece_val = ece.measure(y_prob, y_true)

    return ece_val


# load datasets
pool = Data(unlabeled_data_dir)
pool.setStartState(minimum_training_items)
poolData = pool.data
poolDataIndices = pool.poolDataIndices

validation = Data(validation_data_dir)
validation_data = validation.data
test = Data(test_data_dir)
test_data = test.data

poolDataEmb_train = pool.poolDataEmb
poolDataEmb_val = validation.poolDataEmb
poolDataEmb_test = test.poolDataEmb

training_data = poolData.loc[poolData[iID].isin(pool.indicesKnown)].reset_index(drop=True)
train_data_idx = poolData.index[poolData[iID].isin(pool.indicesKnown)].tolist()
train_data = poolDataEmb_train[train_data_idx]
train_labels = np.array(training_data[goldLabel].tolist())

model.fit(train_data, train_labels)

# Start active learning
sampleIds = []
samplingRanks = []

for alBatch in range(alBatchNum):

    unlabeled_data = poolData.loc[poolData[iID].isin(pool.indicesUnknown)].reset_index(drop=True)
    unlabeled_data_idx = poolData.index[poolData[iID].isin(pool.indicesUnknown)].tolist()
    unl_dataEmb = poolDataEmb_train[unlabeled_data_idx]

    sampledIndices = []
    if al_strategy == 'diversity':
        strategy = DiversitySampling(True)
        sampledItems = strategy.get_model_outliers(poolData, model, unlabeled_data, unl_dataEmb, validation_data,
                                                   poolDataEmb_val, number=alBatchSize)

        for outlier in sampledItems:
            samplingRanks.append(outlier[-1])
            sampleIds.append(outlier[-2])
            sampledIndices.append(outlier[-2])

    elif al_strategy == 'random':
        sampledIndices = random_sampling(pool.indicesUnknown, alBatchSize)
        for i in sampledIndices: sampleIds.append(i)
    elif al_strategy == 'uncertainty':
        sampledIndices = uncertainty_sampling(poolData, model, unl_dataEmb, alBatchSize)
        for i in sampledIndices: sampleIds.append(i)
    else:
        # random sampling by default
        sampledIndices = random_sampling(pool.indicesUnknown, alBatchSize)
        for i in sampledIndices: sampleIds.append(i)

    sampledInd = np.array(sampledIndices)
    pool.indicesKnown = np.concatenate(([pool.indicesKnown, np.array(sampledInd)]))

    pool.indicesUnknown = np.array([])
    for i in poolDataIndices:
        if i not in pool.indicesKnown:
            pool.indicesUnknown = np.concatenate(([pool.indicesUnknown, np.array([i])]));

    training_data = poolData.loc[poolData[iID].isin(pool.indicesKnown)].reset_index(drop=True)
    train_data_idx = poolData.index[poolData[iID].isin(pool.indicesKnown)].tolist()
    train_data = poolDataEmb_train[train_data_idx]
    train_labels = np.array(training_data[goldLabel].tolist())
    # print("Start training.")
    model.fit(train_data, train_labels)

    y_pred_train = model.predict(train_data)
    logits_train = model.predict_proba(train_data)
    probs_train = np.array(logits_train)

    y_pred_val = model.predict(poolDataEmb_val)
    logits_val = model.predict_proba(poolDataEmb_val)
    probs_val = np.array(logits_val)
    val_labels = np.array(validation_data[goldLabel].tolist())

    y_pred_test = model.predict(poolDataEmb_test)
    logits_test = model.predict_proba(poolDataEmb_test)
    probs_test = np.array(logits_test)
    test_labels = np.array(test_data[testGoldLabel].tolist())

    # check if binary or multi class classification
    num_classes = len(set(val_labels))
    if num_classes == 2:
        average = 'binary'
    else:
        average = 'macro'

    sampledItems = ''.join(str(e) + ' ' for e in sampledIndices)

    pre_train, rec_train, f1_train, _ = precision_recall_fscore_support(train_labels, y_pred_train, average=average,
                                                                        beta=1)
    ece_train = ece_score(train_labels, probs_train)
    _, _, f01_train, _ = precision_recall_fscore_support(train_labels, y_pred_train, average=average, beta=0.1)
    _, _, f10_train, _ = precision_recall_fscore_support(train_labels, y_pred_train, average=average, beta=10)

    pre_val, rec_val, f1_val, _ = precision_recall_fscore_support(val_labels, y_pred_val, average=average, beta=1)
    ece_val = ece_score(val_labels, probs_val)
    _, _, f01_val, _ = precision_recall_fscore_support(val_labels, y_pred_val, average=average, beta=0.1)
    _, _, f10_val, _ = precision_recall_fscore_support(val_labels, y_pred_val, average=average, beta=10)

    pre_test, rec_test, f1_test, _ = precision_recall_fscore_support(test_labels, y_pred_test, average=average, beta=1)
    ece_test = ece_score(test_labels, probs_test)
    _, _, f01_test, _ = precision_recall_fscore_support(test_labels, y_pred_test, average=average, beta=0.1)
    _, _, f10_test, _ = precision_recall_fscore_support(test_labels, y_pred_test, average=average, beta=10)

    if average == 'binary':
        brier_train = brier_score_loss(train_labels, probs_train[:, 1])
        brier_val = brier_score_loss(val_labels, probs_val[:, 1])
        brier_test = brier_score_loss(test_labels, probs_test[:, 1])

        print(
            'Iteration: {}. F1: {:1.3f}, Precision: {:1.3f}, Recall: {:1.3f}'.
                format(alBatch, f1_val, pre_val, rec_val))
        # print to result file
        with open(res_path, 'a') as f:
            res_i = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(
                alBatch, sampledItems, pre_train, rec_train, f01_train, f1_train, f10_train, ece_train, brier_train,
                pre_val, rec_val, f01_val, f1_val, f10_val, ece_val, brier_val, pre_test, rec_test, f01_test, f1_test,
                f10_test, ece_test, brier_test)
            f.write(res_i)
    else:

        print(
            'Iteration: {}. F1: {:1.3f}, Precision: {:1.3f}, Recall: {:1.3f}'.
                format(alBatch, f1_val, pre_val, rec_val))
        # print to result file
        with open(res_path, 'a') as f:
            res_i = '{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n'.format(alBatch,
                                                                                                              sampledItems,
                                                                                                              pre_train,
                                                                                                              rec_train,
                                                                                                              f01_train,
                                                                                                              f1_train,
                                                                                                              f10_train,
                                                                                                              ece_train,
                                                                                                              pre_val,
                                                                                                              rec_val,
                                                                                                              f01_val,
                                                                                                              f1_val,
                                                                                                              f10_val,
                                                                                                              ece_val,
                                                                                                              pre_test,
                                                                                                              rec_test,
                                                                                                              f01_test,
                                                                                                              f1_test,
                                                                                                              f10_test,
                                                                                                              ece_test)
            f.write(res_i)

if al_strategy == 'diversity':
    divRanking = pd.DataFrame(
        {'sampleID': sampleIds,
         'diversityRank': samplingRanks
         })

    divRanking.to_csv(resDiversity)


