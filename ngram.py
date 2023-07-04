import math
import pandas as pd
from collections import Counter
import numpy as np

##################################################################
# Vocabulary
##################################################################
# vocabulary function trains with the train.tokens data
def vocabulary():
    trainData = []
    # reads the data from the train data file
    with open("1b_benchmark.train.tokens", "r") as file:
        # appends <START> and <STOP> to each line in the data file
        for line in file:
            tokens = line.split()  # Split the line into tokens
            trainData.append("<START>")
            trainData.extend(tokens) # Add tokens to the list
            trainData.append('<STOP>')
    
    # creates a dictionary 
    # keys are the words in the data, values are the frequency of the words
    tokenDict = Counter(trainData)
    # creates a key and value for <UNK>
    tokenDict["<UNK>"] = 0
 
    # creates a list of duplicate tokens
    duplicateTokens =[]

    # iterates through the dict of tokens
    # if the count < 3 updates UNK and duplicateTokens[]
    for token, count in tokenDict.items():
        if count < 3:
            tokenDict["<UNK>"] += count
            duplicateTokens.append(token)

    # delete duplicate tokens from tokenDict
    for dup in duplicateTokens:
        tokenDict.pop(dup, None)

    return tokenDict

##################################################################
# Unigram Probability
##################################################################
# probability function for the unigram
def word_unigram_probability(tokenDict):
    wordProbDict = {}
    # M is the he total number of tokens in the file including <STOP>
    # not including <START>
    M = sum(tokenDict.values()) - tokenDict['<START>']

    # iterates through every token in the token dictionary
    for token, count in tokenDict.items():
        # calculates the probability of the word based on M
        if token != '<START>':
            wordProbDict[token] = count/M
    return wordProbDict

##################################################################
# Unigram Perplexity
##################################################################
# calculates the perplexity for the unigram
def unigram_perplexity(data, wordProbs):
    log_prob = 0
    M = 0
    # iterates through every word in the data that is not <START>
    for word in data:
       if((word != '<START>')):
           M += 1
           log_prob += math.log(wordProbs.get(word, wordProbs['<UNK>']), 2)

    # H(x) = -1 * ((summation of log probability)/total number of words)
    # the perplexity = 2^(H(x))
    perplexity = math.pow(2, -log_prob / M)
    return perplexity

##################################################################
# Build language model
###################################################################
# builds the language model which is used for bigram and trigram
# data = the input file, n = 2 or 3 (bigram/trigram)
def build_language_model(data_name, n):
    model = []
    # iterates through the data and creates tuples (depending on n)
    with open(data_name, "r") as file:
        for line in file:
            sentence = line.split()
            sentence = ["<START>"] + sentence + ['<STOP>']
            
            for i in range(0, len(sentence)-n+1):
                model.append(tuple(sentence[i: i+n]))
    # creates a dictionaty of the tuples
    # key = the tuple | value = the frequency of the tuple
    tokenizedModel = Counter(model)
    return tokenizedModel


###################################################################
# Bigram Perplexity
###################################################################
# calculates the perplexity for the bigram 
def bigram_perplexity(data_name, bigramModel, unigramTokens):
    log_prob = 0
    M = 0
    with open(data_name, "r") as file:
        for line in file:
            tokens = line.split()
            tokens = ["<START>"] + tokens + ['<STOP>']

            data_language_model = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

            for tup in data_language_model:
                val = 0
                
                tuple_prob = bigramModel.get(tup, 0)
                unigram_prob = unigramTokens.get(tup[0],0)
                
                if unigram_prob != 0:
                    val = float(tuple_prob) / float(unigram_prob)
                    
                if(val > 0):
                    log_prob += math.log(val, 2)
            M += len(tokens) - 1
    perplexity = math.pow(2, -log_prob / M)
    return perplexity

   
###################################################################
# Trigram Perplexity
###################################################################
# calculates the perplexity for the trigram 
def trigram_perplexity(data_name, trigramModel, bigramModel, unigramTokens):
    log_prob = 0
    M = 0
    with open(data_name, "r") as file:
        for line in file:
            tokens = line.split()
            tokens = ["<START>"] + tokens + ['<STOP>']

            data_language_model = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]
            
            for tup in data_language_model:
                val = 0
                unigram_prob = 0

                tri_prob = trigramModel.get(tup, 0)

                if(tup[0] == '<START>'):
                    unigram_prob = unigramTokens.get(tup[0], 0)
                else:
                    unigram_prob = bigramModel.get(tup[0:2],0)

                if unigram_prob != 0:
                    val = float(tri_prob) / float(unigram_prob)

                if(val > 0):
                    log_prob += math.log(val, 2)
            M += len(tokens) - 1

    perplexity = math.pow(2, -log_prob / M)
    return perplexity

###################################################################
# AS Unigram Perplexity
###################################################################
def AS_unigram_perplexity(data, wordProbs, alpha):
    log_prob = 0
    M = 0
    V = len(wordProbs) # total number of unique words
    N = sum(wordProbs.values())
    for word in data:
       if((word != '<START>')):
           M += 1
        #    prob = (wordProbs.get(word, 0) + alpha) / (N + alpha * V)
           prob = (wordProbs.get(word, wordProbs['<UNK>']) + alpha) / (N + alpha * V)
        #    prob = (wordProbs.get(word, wordProbs['<UNK>']))
           log_prob += math.log(prob, 2) 
        #    log_prob += math.log(wordProbs.get(word, wordProbs['<UNK>']), 2)

    # H(x) = -1 * ((summation of log probability)/total number of words)
    # the perplexity = 2^(H(x))
    perplexity = math.pow(2, -log_prob / M)
    return perplexity

###################################################################
# AS Bigram Perplexity
###################################################################
def AS_bigram_perplexity(data_name, bigramModel, unigramTokens, alpha):
    log_prob = 0
    M = 0
    V = len(wordProbs)
    with open(data_name, "r") as file:
        for line in file:
            tokens = line.split()
            tokens = ["<START>"] + tokens + ['<STOP>']

            data_language_model = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

            for tup in data_language_model:
                val = 0
                
                tuple_prob = bigramModel.get(tup, 0) + alpha
                unigram_prob = unigramTokens.get(tup[0],0) + (alpha * V)
                
                if unigram_prob != 0:
                    val = float(tuple_prob) / float(unigram_prob)
                    
                if(val > 0):
                    log_prob += math.log(val, 2)
            M += len(tokens) - 1
    perplexity = math.pow(2, -log_prob / M)
    return perplexity


###################################################################
# AS Trigram Perplexity
###################################################################
def AS_trigram_perplexity(data_name, trigramModel, bigramModel, unigramTokens, alpha):
    log_prob = 0
    M = 0
    V = len(wordProbs)
    with open(data_name, "r") as file:
        for line in file:
            tokens = line.split()
            tokens = ["<START>"] + tokens + ['<STOP>']

            data_language_model = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]
            
            for tup in data_language_model:
                val = 0
                unigram_prob = 0

                tri_prob = trigramModel.get(tup, 0) 

                if(tup[0] == '<START>'):
                    unigram_prob = unigramTokens.get(tup[0], 0) 
                else:
                    unigram_prob = bigramModel.get(tup[0:2],0) 

                if unigram_prob != 0:
                    val = (float(tri_prob) + alpha) / (float(unigram_prob) + (alpha * V))

                if(val > 0):
                    log_prob += math.log(val, 2)
            M += len(tokens) - 1

    perplexity = math.pow(2, -log_prob / M)
    return perplexity

###################################################################
# Smoothing with Linear Interpolation
###################################################################

def linear_interpolation(data, lambda1, lambda2, lambda3, tokenDict, bigram_model):
    model = []
    with open("1b_benchmark.train.tokens", "r") as file:
        for line in file:
            sentence = line.split()
            sentence = ["<START>"] + ["<START>"] + sentence + ['<STOP>']
            
            for i in range(0, len(sentence)-2):
                model.append(tuple(sentence[i: i+3]))

    trigram_model = Counter(model)

    theta = 0
    M = 0
    tokens_without_start = sum(tokenDict.values()) - tokenDict['<START>']
    
    with open(data, "r") as file:
        for line in file:
            tokens = line.split()
            tokens = ["<START>"]  + ["<START>"] + tokens + ['<STOP>']

            data_language_model = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]
            
            for tup in data_language_model:
                val = 0

                if (tup[0] == '<START>') and (tup[1] == '<START>'):
                    unigram_prob = float(tokenDict[tup[2]]/tokens_without_start)
                    bigram_prob = float(bigram_model[tup[1:3]]) / float(tokenDict[tup[1:3][0]])
                    trigram_prob = float(trigram_model[tup]) / float(tokenDict[tup[0]])
                    val += lambda1 * unigram_prob
                    val += lambda2 * bigram_prob
                    val += lambda3 * trigram_prob
                else:
                    unigram_prob = float(tokenDict[tup[2]]/tokens_without_start)
                    bigram_unigram_prob = float(tokenDict[tup[1:3][0]])
                    if (bigram_unigram_prob != 0):
                        bigram_prob = float(bigram_model[tup[1:3]]) / float(tokenDict[tup[1:3][0]])
                    else:
                        bigram_prob = 0
                    trigram_bigram_prob = float(bigram_model[tup[0:2]])
                    if(trigram_bigram_prob != 0):
                        trigram_prob = float(trigram_model[tup]) / float(bigram_model[tup[0:2]])
                    else:
                        trigram_prob = 0
                    val += lambda1 * unigram_prob
                    val += lambda2 * bigram_prob
                    val += lambda3 * trigram_prob
                if(val > 0):
                    theta += math.log(val, 2)
            M += len(tokens) - 2

    perplexity = math.pow(2, -theta / M)
    return perplexity



#*************************************************************************************************
# Part 1: N-Gram Language Modeling
#*************************************************************************************************
print("\n ---- Part 1:  N-Gram Language Modeling ---- \n")
#
#
###################################################################
# TRAIN DATA for Unigram, Bigram and Trigram
###################################################################
# Read training tokens file
train_data = []
train_name = "1b_benchmark.train.tokens"
with open("1b_benchmark.train.tokens", "r") as file:
    # appends <START> and <STOP> to each line in the data file
    for line in file:
        tokens = line.split()  # Split the line into tokens
        # train_data.extend(tokens)  # Add tokens to the list
        train_data.append("<START>")
        train_data.extend(tokens)
        train_data.append('<STOP>')

# tokenize the input file
# key = word | value = frequency of that word
tokenDict = {}
tokenDict = vocabulary()

# Build unigram model
# the probability of each word in the tokens
wordProbs = {}
wordProbs = word_unigram_probability(tokenDict)

# Evaluate unigram model for training data
unigram_log_prob_train = unigram_perplexity(train_data, wordProbs)
print("Unigram Log Probability (Train):", unigram_log_prob_train)

# Build bigram model
bigram_model = build_language_model(train_name, 2)

# # Evaluate bigram model
bigram_log_prob_train = bigram_perplexity(train_name, bigram_model, tokenDict)
print("Bigram Log Probability (Train):", bigram_log_prob_train)

# Build trigram model
trigram_model = build_language_model(train_name, 3)

# Evaluate trigram model
trigram_log_prob_train = trigram_perplexity(train_name, trigram_model, bigram_model, tokenDict)
print("Trigram Log Probability (Train):", trigram_log_prob_train)

print("\n")


###################################################################
# TEST DATA for Unigram, Bigram and Trigram
###################################################################
test_data = []
test_name = "1b_benchmark.test.tokens"
train_name  = "1b_benchmark.test.tokens"
with open("1b_benchmark.test.tokens", "r") as file:
    for line in file:
        tokens = line.split()  # Split the line into tokens
        test_data.append("<START>")
        test_data.extend(tokens)  # Add tokens to the list
        test_data.append('<STOP>')

# Evaluate unigram model
unigram_log_prob_test = unigram_perplexity(test_data, wordProbs)
print("Unigram Log Probability (Test):", unigram_log_prob_test)

# Evaluate bigram model
bigram_log_prob_test = bigram_perplexity(test_name, bigram_model, tokenDict)
print("Bigram Log Probability (Test):", bigram_log_prob_test)

# # Evaluate trigram model
trigram_log_prob_test = trigram_perplexity(test_name,  trigram_model, bigram_model, tokenDict)
print("Trigram Log Probability (Test):", trigram_log_prob_test)

print("\n")

###################################################################
# DEV DATA for Unigram, Bigram and Trigram
###################################################################
dev_data = []
dev_name = "1b_benchmark.dev.tokens"
with open("1b_benchmark.dev.tokens", "r") as file:
    for line in file:
        tokens = line.split()  # Split the line into tokens
        dev_data.append("<START>")
        dev_data.extend(tokens)  # Add tokens to the list
        dev_data.append('<STOP>')

# Evaluate unigram model
unigram_log_prob_dev = unigram_perplexity(dev_data, wordProbs)
print("Unigram Log Probability (Dev):", unigram_log_prob_dev)

# # Evaluate bigram model
bigram_log_prob_dev = bigram_perplexity(dev_name, bigram_model, tokenDict)
print("Bigram Log Probability (Dev):", bigram_log_prob_dev)

# # Evaluate trigram model
trigram_log_prob_dev = trigram_perplexity(dev_name, trigram_model, bigram_model, tokenDict)
print("Trigram Log Probability (Dev):", trigram_log_prob_dev)

print("\n")

##################################################################
# HDTV DATA for Unigram, Bigram and Trigram
##################################################################

hdtv_data = []
hdtv_name = "hdtv.txt"
M = 0
with open("hdtv.txt", "r") as file:
    for line in file:
        tokens = line.split()  # Split the line into tokens
        hdtv_data.append("<START>")
        hdtv_data.extend(tokens)  # Add tokens to the list
        hdtv_data.append('<STOP>')
        
# Evaluate unigram model
unigram_log_prob_hdtv = unigram_perplexity(hdtv_data, wordProbs)
print("Unigram Log Probability (HDTV):", unigram_log_prob_hdtv)

# Evaluate bigram model
bigram_log_prob_hdtv = bigram_perplexity(hdtv_name, bigram_model, tokenDict)
print("Bigram Log Probability (HDTV):", bigram_log_prob_hdtv)

# # Evaluate trigram model
trigram_log_prob_hdtv = trigram_perplexity(hdtv_name, trigram_model, bigram_model, tokenDict)
print("Trigram Log Probability (HDTV):", trigram_log_prob_hdtv)

print("\n\n")

#*************************************************************************************************
# Part 2: Additive Smoothing for Unigram, Bigram, Trigram
#*************************************************************************************************
print("---- Part 2: Additive Smoothing for Unigram, Bigram, Trigram ---- \n")
#
#
###################################################################
# TRAIN DATA for Unigram, Bigram and Trigram (alpha = 1)
###################################################################
alpha = 1
print("Alpha =", alpha)
unigram_perplexity_as_traindata = AS_unigram_perplexity(train_data, wordProbs, alpha)
print("Unigram Additive Smoothing (Train)", unigram_perplexity_as_traindata)

bigram_perplexity_as_traindata = AS_bigram_perplexity(train_name, bigram_model, tokenDict, alpha)
print("Bigram Additive Smoothing (Train)", bigram_perplexity_as_traindata)

trigram_perplexity_as_traindata = AS_trigram_perplexity(train_name, trigram_model, bigram_model, tokenDict, alpha)
print("Trigram Additive Smoothing (Train)", trigram_perplexity_as_traindata)

print("\n")
###################################################################
# Dev DATA for Unigram, Bigram and Trigram (alpha = 1)
###################################################################
unigram_perplexity_as_dev = AS_unigram_perplexity(dev_data, wordProbs, alpha)
print("Unigram Additive Smoothing (Dev)", unigram_perplexity_as_dev)

bigram_perplexity_as_dev = AS_bigram_perplexity(dev_name, bigram_model, tokenDict, alpha)
print("Bigram Additive Smoothing (Dev)", bigram_perplexity_as_dev)

trigram_perplexity_as_dev = AS_trigram_perplexity(dev_name, trigram_model, bigram_model, tokenDict, alpha)
print("Trigram Additive Smoothing (Dev)", trigram_perplexity_as_dev)

print("\n")

###################################################################
# Test DATA for Unigram, Bigram and Trigram (alpha = 1)
###################################################################
unigram_perplexity_as_test = AS_unigram_perplexity(test_data, wordProbs, alpha)
print("Unigram Additive Smoothing (Test)", unigram_perplexity_as_test)

bigram_perplexity_as_test = AS_bigram_perplexity(test_name, bigram_model, tokenDict, alpha)
print("Bigram Additive Smoothing (Test)", bigram_perplexity_as_test)

trigram_perplexity_as_test = AS_trigram_perplexity(test_name, trigram_model, bigram_model, tokenDict, alpha)
print("Trigram Additive Smoothing (Test)", trigram_perplexity_as_test)

print("\n")

# ###################################################################
# # HDTV DATA for Unigram, Bigram and Trigram (alpha = 1)
# ###################################################################
# unigram_perplexity_as_hdtv = AS_unigram_perplexity(hdtv_data, wordProbs, alpha)
# print("Unigram Additive Smoothing (HDTV)", unigram_perplexity_as_hdtv)

# bigram_perplexity_as_hdtv = AS_bigram_perplexity(hdtv_name, bigram_model, tokenDict, alpha)
# print("Bigram Additive Smoothing (HDTV)", bigram_perplexity_as_hdtv)

# trigram_perplexity_as_hdtv = AS_trigram_perplexity(hdtv_name, trigram_model, bigram_model, tokenDict, alpha)
# print("Trigram Additive Smoothing (HDTV)", trigram_perplexity_as_hdtv)

# print("\n")

#*************************************************************************************************
# Part 3: Smoothing With Linear Interpolation 
#*************************************************************************************************
print("---- Part 3: Smoothing With Linear Interpolation  ---- \n")
#
#
lambda1 = 0.1
lambda2 = 0.3
lambda3 = 0.6
linear_inter_train = linear_interpolation(train_name, lambda1, lambda2, lambda3, tokenDict, bigram_model)
print("Smoothing with linear interpolation (Train) with lambdas (", lambda1, ",", lambda2, ",", lambda3, ") =", linear_inter_train)

linear_inter_dev = linear_interpolation(dev_name, lambda1, lambda2, lambda3, tokenDict, bigram_model)
print("Smoothing with linear interpolation (Dev) with lambdas (", lambda1, ",", lambda2, ",", lambda3, ") =", linear_inter_dev)

linear_inter_test = linear_interpolation(test_name, lambda1, lambda2, lambda3, tokenDict, bigram_model)
print("Smoothing with linear interpolation (Test) with lambdas (", lambda1, ",", lambda2, ",", lambda3, ") =", linear_inter_test)

linear_inter_hdtv = linear_interpolation(hdtv_name, lambda1, lambda2, lambda3, tokenDict, bigram_model)
print("Smoothing with linear interpolation (HDTV) with lambdas (", lambda1, ",", lambda2, ",", lambda3, ") =", linear_inter_hdtv)

print("\n")
