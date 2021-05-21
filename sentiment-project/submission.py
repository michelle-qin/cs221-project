#!/usr/bin/python

import random
from typing import Callable, Dict, List, Tuple, TypeVar

from util import *

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]

############################################################
# Problem 3: binary classification
############################################################

############################################################
# Problem 3a: feature extraction


def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    word_dict = dict()
    delimited_x = x.split(" ")
    for word in delimited_x:
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1
    return word_dict

    #raise Exception("Not implemented yet")
    # END_YOUR_CODE


############################################################
# Problem 3b: stochastic gradient descent

T = TypeVar('T')


def learnPredictor(trainExamples: List[Tuple[T, int]],
                   validationExamples: List[Tuple[T, int]],
                   featureExtractor: Callable[[T], FeatureVector],
                   numEpochs: int, eta: float) -> WeightVector:
    '''
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Notes: 
    - Only use the trainExamples for training!
    - You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. 
    - The identity function may be used as the featureExtractor function during testing.
    - The predictor should output +1 if the score is precisely 0.
    '''
    def predictor(i):
            product = dotProduct(weights, featureExtractor(i))
            if product >= 0:
                return 1
            else: 
                return -1
    weights = {}  # feature => weight
    for t in range(numEpochs):
        for train in trainExamples:
            if dotProduct(weights, featureExtractor(train[0])) * train[1] < 1:
                increment(weights, eta * train[1], featureExtractor(train[0]))
        evaluatePredictor(trainExamples, predictor)
        evaluatePredictor(validationExamples, predictor)
    return weights


    # BEGIN_YOUR_CODE (our solution is 13 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    # END_YOUR_CODE
    #return weights


############################################################
# Problem 3c: generate test case


def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''
    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) should be a dict whose keys are a subset of the keys in weights
    # and values can be anything (randomize!) with a score for the given weight vector.
    # y should be 1 or -1 as classified by the weight vector.
    # y should be 1 if the score is precisely 0.

    # Note that the weight vector can be arbitrary during testing.
    def generateExample() -> Tuple[Dict[str, int], int]:
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}
        key = random.choices(list(weights), k=3) # why k = 3
        for i in range(3):
            phi[key[i]] = random.random() #why any random number
        product = dotProduct(phi, weights) #does the dot product assume 0
        if product >=0: # how to decide how many to choose
            y = 1 
        else:
            y = -1

        # raise Exception("Not implemented yet")
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 3e: character features


def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    '''
    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        word_dict = dict()
        x = ''.join(x.split())
        #print(x)
        for i in range(len(x) - n + 1):
            word = x[i:i+n]
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
            
        return word_dict
        #raise Exception("Not implemented yet")
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f:


def testValuesOfN(n: int):
    '''
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    '''
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples,
                             validationExamples,
                             featureExtractor,
                             numEpochs=20,
                             eta=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights,
                        'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(
        trainExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(
        validationExamples, lambda x:
        (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" %
           (trainError, validationError)))


############################################################
# Problem 4: k-means
############################################################




def kmeans(examples: List[Dict[str, float]], K: int,
           maxEpochs: int) -> Tuple[List, List, float]:
    '''
    examples: list of examples, each example is a string-to-float dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxEpochs: maximum number of epochs to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j),
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 28 lines of code, but don't worry if you deviate from this)
    mu = random.choices(examples, k=K) # randomize, pick k mu's
    assignments = [None] * len(examples)
    prev_assignments = [None] * len(examples)
    distances = [None] * len(examples)
    examples_2 = [None] * len(examples)
    mu_2 = [None] * len(mu)
    
    for i in range(len(examples)): #calculating the square
        example = examples[i]
        examples_2[i] = dotProduct(example, example)

    for t in range(maxEpochs):
        assignments = [None] * len(examples) #map examples to centroid/mu
        mu_2 = [None] * len(mu)
        for i in range(len(mu)): #calculating the square for each mu
            mu_i = mu[i]
            mu_2[i] = dotProduct(mu_i, mu_i)
        
        for i in range(len(examples)): #go through each example
            example = examples[i]
            distances[i] = sys.maxsize
            assignments[i] = 0
            for j in range(len(mu)): #go through each mu
                z_i = mu_2[j] + examples_2[i] - 2 * dotProduct(example, mu[j])
                if z_i < distances[i]:
                    assignments[i] = j # assign to mu index
                    distances[i] = z_i
        
        if assignments == prev_assignments:
            break
        prev_assignments = assignments

        for i in range(len(mu)):
            mu_updated = { }
            mu_entries = []
            for x in range(len(examples)):
                if mu[i] == mu[assignments[x]]:
                    mu_entries.append(x)
            for j in range(len(mu_entries)):
                increment(mu_updated, 1, examples[mu_entries[j]])
            count = len(mu_entries)
            for k in mu_updated:
                mu_updated[k] = mu_updated[k]/count
            mu[i] = mu_updated 
           
    loss = 0
    for i in range(len(examples)): #calculating the square
        loss += mu_2[assignments[i]] + examples_2[i] - 2 * dotProduct(examples[i], mu[assignments[i]])

    return mu, assignments, loss

    

