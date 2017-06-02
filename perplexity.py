import sys
import csv
import numpy as np
import string

# Laplace Unigram
def P_LU(unigram):
    if unigram in unigrams.keys():
        result = (unigrams[unigram] + 1) / (tokenCount + len(unigrams) + 1)
    else:
        result = 1 / (len(unigrams) + 1)
    return result

# Laplace Bigram
def P_LB(bigram):
    if bigram in bigrams.keys():
        result = (bigrams[bigram] + 1)/(unigrams[bigram[0]] + len(unigrams) + 1)
    else:
        result = 1 / (len(unigrams) + 1)
    return result

# Interpolated Bigram
def P_IB(bigram):
    if bigram in bigrams.keys():
        b_result = lamda * (bigrams[bigram] / unigrams[bigram[0]])
        u_result = (1 - lamda) * (1 + unigrams[bigram[1]])
        u_result /= (tokenCount + len(unigrams) + 1)
        result = b_result + u_result
    elif bigram[1] in unigrams.keys():
        result = (1 - lamda) * (1 + unigrams[bigram[1]])
        result /= (len(unigrams) + tokenCount + 1)
    else:
        result = (1 - lamda) / (len(unigrams) + tokenCount + 1)
    return result

if __name__ == '__main__':
    # intialize core variables
    if len(sys.argv) == 4:
        bigramFile = str(sys.argv[1])
        unigramFile = str(sys.argv[2])
        testFile = str(sys.argv[3])
    else:
        print("<Using default parameters>")
        bigramFile = 'bigram.lm'
        unigramFile = 'unigram.lm'
        testFile = 'test.txt'

    print("Perplexities in \"" + testFile + "\" with selected language model")

    # Core variables
    fileTokens = []
    tokenCount = 0
    unigrams = {}
    bigrams = {}
    lamda = 0.3 # Determined with perplexity.py

    # read unigram.lm
    with open('unigram.lm', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        headerCaught = False
        header = ""
        for row in csvreader:
            if headerCaught == False: # Save header row.
                header = row
                headerCaught = True
                continue
            word, occurrences = row
            unigrams[word] = int(occurrences)
            tokenCount += int(occurrences)

    # read bigram.lm
    with open('bigram.lm', 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        headerCaught = False
        header = ""
        for row in csvreader:
            if headerCaught == False: # Save header row.
                header = row
                headerCaught = True
                continue
            w1, w2, occurrences, MLE, Laplace, Inter, Katz = row
            two_word = (w1, w2)
            bigrams[two_word] = int(occurrences)

    # Read the test file
    fileLine = ""
    f = open(testFile, 'r')
    for line in f:
        fileLine = fileLine + line.replace("\n", " ")
    sentences = fileLine.split(".")
    for sentence in sentences:
        if(len(sentence) == 0):
            continue
        while len(sentence) > 0 and sentence[0] == ' ':
            sentence = sentence[1:]
        sentence = sentence.lower()
        sentence = "<s> " + sentence + " </s>"
        sentenceTokens = sentence.split(" ")
        for token in sentenceTokens:
            token = token.strip('[]?:!.,;\"\'_')
            token = token.rstrip('[]?:!.,;\"\'_')
            if len(token) > 0:
                fileTokens.append(token)

    # Calculate Perplexities
    N = len(fileTokens)

    # get Laplace unigram Perplexity
    SumOfLogs = 0
    for word in fileTokens:
        SumOfLogs +=  np.log2( P_LU(word) )
    Laplace_unigrams_PP = 2**((-1/N) * SumOfLogs)
    print("Laplace Unigram = " + str(Laplace_unigrams_PP))

    # get Laplace bigram Perplexity
    SumOfLogs = 0
    for i in range(1,len(fileTokens)-1):
        words = (fileTokens[i-1], fileTokens[i])
        SumOfLogs +=  np.log2( P_LB(words) )
    Laplace_bigrams_PP = 2**((-1/N) * SumOfLogs)
    print("Laplace Bigram = " + str(Laplace_bigrams_PP))

    # get Interpolated bigram Perplexity
    SumOfLogs = 0
    for i in range(1,len(fileTokens)-1):
        words = (fileTokens[i-1], fileTokens[i])
        SumOfLogs +=  np.log2( P_IB(words) )
    Inter_bigrams_PP = 2**((-1/N) * SumOfLogs)
    print("Interpolated Bigram = " + str(Inter_bigrams_PP))
