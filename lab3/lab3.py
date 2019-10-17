#!/usr/bin/python
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting
import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random

# ## Bayes classifier

# in: labels - N vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses,1))

    for idc, c in enumerate(classes):
        idp = np.where(labels == c)[0]   # Ids of points in the given class
        prior[idc] = np.sum(W[idp]) / np.sum(W)

    return prior

# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))

    for idx, c in enumerate(classes):
        idy = np.where(labels == c)[0]   # Get all vectors corresponding to the class
        w_matrix  = X[idy] * W[idy]   # Multiply all vectors by their weights
        mu[idx] = w_matrix.sum(axis=0) / np.sum(W[idy])   # Sum up and divide by the sum of vector weights

    for idx, c in enumerate(classes):
        idy = np.where(labels == c)[0]   # Get all vectors for the class
        variances = (X[idy] - mu[idx]) ** 2   # Calculate the variances for each vector
        w_variances = W[idy] * variances   # Multiply variances by the weights
        mean = np.sum(w_variances, axis=0) / np.sum(W[idy])   # Sum and divide by the sum of weights
        sigma[idx] = np.diag(mean)  # Get the diagonal matrix for Naive Bayes

    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    for idc in range(Nclasses):
        ln_sigma = - np.log(np.linalg.det(sigma[idc])) / 2
        diff = X - mu[idc]
        ln_prior = np.log(prior[idc])
        for point in range(Npts):
            logProb[idc][point] = ln_sigma - np.inner(diff[point] / np.diag(sigma[idc]), diff[point]) / 2 + ln_prior

    # one possible way of finding max a-posteriori once
    # you have computed the log posterior
    h = np.argmax(logProb,axis=0)
    return h


# The implemented functions can now be summarized into the `BayesClassifier` class, which we will use later to test the classifier, no need to add anything else here:

class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)


# ## Test the Maximum Likelihood estimates
# 
# Call `genBlobs` and `plotGaussian` to verify your estimates.


X, labels = genBlobs(centers=8)
mu, sigma = mlParams(X,labels)
# classifyBayes(X, computePrior(labels), mu, sigma)
# plotGaussian(X,labels,mu,sigma)


# Call the `testClassifier` and `plotBoundary` functions for this part.


# testClassifier(BayesClassifier(), dataset='iris', split=0.7)
# plotBoundary(BayesClassifier(), dataset='iris',split=0.7)

# testClassifier(BayesClassifier(), dataset='vowel', split=0.7)
# plotBoundary(BayesClassifier(), dataset='vowel',split=0.7)


# ## Boosting functions to implement
# 
# The lab descriptions state what each function should do.


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    # these will come in handy later on
    Npts,Ndims = np.shape(X)

    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list

    # The weights for the first iteration
    wCur = np.ones((Npts,1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)

        # Error = sum of weights - weights of correctly classified instances
        error = np.sum(wCur)
        c_votes = np.where(vote == labels)[0]   # correct votes
        for vote in c_votes:
            error -= wCur[vote]

        alpha = (np.log(1 - error) - np.log(error)) / 2
        alphas.append(alpha)   # append the new alpha

        f_votes = np.where(vote != labels)[0]   # incorrect votes
        wOld = wCur

        # Weights for correctly classified * e^{-alpha} to decrease the weight
        for vote in c_votes:
            wCur[vote] = wOld[vote] * np.exp(-alpha)

        # Weights for misclassified * e^{alpha} to increase the weight
        for vote in f_votes:
            wCur[vote] = wOld[vote] * np.exp(alpha)

        wCur /= np.sum(wCur)

    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # here we can do it by filling in the votes vector with weighted votes
        for idx, classifier in enumerate(classifiers):
            step = classifier.classify(X)
            for point in range(Npts):
                votes[point][step[point]] += alphas[idx]

        # one way to compute yPred after accumulating the votes
        return np.argmax(votes, axis=1)


# The implemented functions can now be summarized another classifer, the `BoostClassifier` class.
# This class enables boosting different types of classifiers by initializing it with the `base_classifier` argument.
# No need to add anything here.


class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# ## Run some experiments
# 
# Call the `testClassifier` and `plotBoundary` functions for this part.


# testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='iris',split=0.7)

# plotBoundary(BoostClassifier(BayesClassifier()), dataset='iris',split=0.7)


# testClassifier(BoostClassifier(BayesClassifier(), T=10), dataset='vowel',split=0.7)
# plotBoundary(BoostClassifier(BayesClassifier()), dataset='vowel',split=0.7)




# Now repeat the steps with a decision tree classifier.


# testClassifier(DecisionTreeClassifier(), dataset='iris', split=0.7)
# plotBoundary(DecisionTreeClassifier(), dataset='iris', split=0.7)

# testClassifier(DecisionTreeClassifier(), dataset='vowel', split=0.7)
# plotBoundary(DecisionTreeClassifier(), dataset='vowel', split=0.7)

# testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)
# plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='iris',split=0.7)

testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)
plotBoundary(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='vowel',split=0.7)



# ## Bonus: Visualize faces classified using boosted decision trees
# 
# Note that this part of the assignment is completely voluntary! First, let's check how a boosted decision tree classifier performs on the olivetti data. Note that we need to reduce the dimension a bit using PCA, as the original dimension of the image vectors is `64 x 64 = 4096` elements.


#testClassifier(BayesClassifier(), dataset='olivetti',split=0.7, dim=20)



#testClassifier(BoostClassifier(DecisionTreeClassifier(), T=10), dataset='olivetti',split=0.7, dim=20)


# You should get an accuracy around 70%. If you wish, you can compare this with using pure decision
# trees or a boosted bayes classifier. Not too bad, now let's try and classify a face as belonging
# to one of 40 persons!


#X,y,pcadim = fetchDataset('olivetti') # fetch the olivetti data
#xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,0.7) # split into training and testing
#pca = decomposition.PCA(n_components=20) # use PCA to reduce the dimension to 20
#pca.fit(xTr) # use training data to fit the transform
#xTrpca = pca.transform(xTr) # apply on training data
#xTepca = pca.transform(xTe) # apply on test data
# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
#classifier = BoostClassifier(DecisionTreeClassifier(), T=10).trainClassifier(xTrpca, yTr)
#yPr = classifier.classify(xTepca)
# choose a test point to visualize
#testind = random.randint(0, xTe.shape[0]-1)
# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
#visualizeOlivettiVectors(xTr[yTr == yPr[testind],:], xTe[testind,:])

