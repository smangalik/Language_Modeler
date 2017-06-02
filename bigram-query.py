import sys
import csv
import string

# Absolute Discounting Bigram
def P_ADB(bigram):
    result = (bigrams[bigram] - 0.5) / (unigrams[bigram[0]])
    return result

# Laplace Unigram
def P_LU(unigram):
    if unigram in unigrams.keys():
        result = (unigrams[unigram] + 1) / (tokenCount + len(unigrams) + 1)
    else:
        result = 1 / (len(unigrams) + 1)
    return result

if __name__ == '__main__':
    # intialize arguments
    if len(sys.argv) == 6:
        bigramFile = str(sys.argv[1])
        unigramFile = str(sys.argv[2])
        x = sys.argv[3].lower()
        y = sys.argv[4].lower()
        smoothing = str(sys.argv[5]).upper()
    else:
        print("<Using default parameters>")
        bigramFile = 'bigram.lm'
        unigramFile = 'unigram.lm'
        x = 'the'
        y = 'fly'
        smoothing = 'M'

    # Core variables
    tokenCount = 0
    unigrams = {}
    bigrams = {}
    MLE_bigrams = {}
    Laplace_bigrams = {}
    Inter_bigrams = {}
    Katz_bigrams = {}
    lamda = 0.3

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
    if x not in unigrams.keys():
        print("\"" + x + "\" was not in the training set, "
            + "there are no useful bigrams estimates")
        exit()

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
            MLE_bigrams[two_word] = float(MLE)
            Laplace_bigrams[two_word] = float(Laplace)
            Inter_bigrams[two_word] = float(Inter)
            Katz_bigrams[two_word] = float(Katz)

    # get results
    value = 0
    if smoothing == 'M':
        print("Using MLE, P(" + y + "|" + x + ") = ", end="")
        if (x,y) in bigrams.keys():
            value = MLE_bigrams[(x,y)]
        else:
            value = 0
    elif smoothing == 'L':
        print("Using Laplace, P(" + y + "|" + x + ") = ", end="")
        if (x,y) in bigrams.keys():
            value = Laplace_bigrams[(x,y)]
        else:
            value = 1 / (unigrams[x] + len(unigrams) + 1)
    elif smoothing == 'I':
        print("Using Interpolated, P(" + y + "|" + x + ") = ", end="")
        if (x,y) in bigrams.keys():
            value = Inter_bigrams[(x,y)]
        else:
            value = (1 - lamda) * (1 + unigrams[word])
            value /= (len(unigrams) + tokenCount + 1)
    elif smoothing == 'K':
        print("Using Katz-BackOff, P(" + y + "|" + x + ") = ", end="")
        if (x,y) in bigrams.keys():
            value = Katz_bigrams[(x,y)]
        else:
            aSum = 0
            bSum = 0
            for bigram, count in bigrams.items():
                if bigram[0] == x:
                    aSum += P_ADB(bigram)
                else:
                    bSum += P_LU(bigram[1])
            alpha = 1 - aSum
            beta = P_LU(word) / bSum
            value = alpha * beta
    else:
        print("Invalid Probability Function, try [M, L, I, K]")
        exit()

    print(value)
