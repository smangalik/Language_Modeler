import sys
import csv
import string

# MLE Probability: P(y|x) = #(x,y)/#(y)
def MLE_Probability(x_y, x, y):
    result = bigrams[x_y] / unigrams[x]
    return round(result, 4)

# Interpolated Smoothing P(y|x) = (lamda)MLE(y|x) + (1 - lamda)Laplace(y)
def Inter_Probability(x_y, x, y):
    b_result = lamda * (bigrams[x_y] / unigrams[x])
    u_result = (1 - lamda) * (1 + unigrams[y])
    u_result /= (len(words) + len(unique_words) + 1)
    result = b_result + u_result
    return round(result, 4)

# Absolute Discounting P(y|x) = #(x_y)-0.5 / #(x)
def AD_Probability(x_y, x, y):
    result = (bigrams[x_y] - 0.5) / (unigrams[x])
    return round(result, 4)

# Laplace Smoothing: P(y|x) = #(x,y) + 1/#(y)+(vocab size + 1)
def Laplace_Probability(x_y, x, y):
    result = (bigrams[x_y] + 1)/(unigrams[x] + len(unique_words) + 1)
    return round(result, 4)

if __name__ == '__main__':
    # intialize core variables
    if len(sys.argv) == 1:
        trainingFile = 'train.txt'
    else:
        trainingFile = str(sys.argv[1])
    sentences = []
    sentence_words = []
    words = []
    twowords = []
    fileLine = ""
    unigrams = {}
    bigrams = {}
    laplace_bigrams = {}
    lamda = 0.3 # Determined with perplexity.py

    # text as a single line
    f = open(trainingFile, 'r')
    for line in f:
        fileLine = fileLine + line.replace("\n", " ")
    filter(None, sentences)
    sentences = fileLine.split(".")
    for sentence in sentences:
        if(len(sentence) == 0):
            continue
        while len(sentence) > 0 and sentence[0] == ' ':
            sentence = sentence[1:]
        sentence = sentence.lower()
        sentence = "<s> " + sentence + " </s>"
        sentence_words.append(sentence.split(" "))

    # get all unique words
    words = [word for sentence in sentence_words for word in sentence]
    words = [x.rstrip('[]?:!.,;"\'_') for x in words if x]
    words = [x.strip('[]?:!.,;"\'_') for x in words]
    wordset = set(words)
    unique_words = list(wordset)

    # get all unique bigrams
    for i in range(len(words)-1):
        if words[i] != '' and words[i+1] != '':
            twowords.append(words[i] + ' ' + words[i+1])
    twoword_set = set(twowords)
    unique_twowords = list(twoword_set)

    # write unigram.lm
    for unique in unique_words:
        unigrams[unique] = words.count(unique)
    with open('unigram.lm', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
        writer.writerow(['word', 'occurrences'])
        for key, value in unigrams.items():
            if key == '':
                continue
            writer.writerow([key, value])

    # write bigram.lm
    for unique_twoword in unique_twowords:
        bigrams[unique_twoword] = twowords.count(unique_twoword)
    with open('bigram.lm', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
        writer.writerow(['word1', 'word2', 'occurrences', 'MLE'
            ,'Laplace', 'Interpolated', 'AD'])
        for key, value in bigrams.items():
            split_words = key.split(' ')
            MLE = MLE_Probability(key, split_words[0], split_words[1])
            Laplace = Laplace_Probability(key, split_words[0], split_words[1])
            Inter = Inter_Probability(key, split_words[0], split_words[1])
            AD = AD_Probability(key, split_words[0], split_words[1])

            laplace_bigrams[key] = unigrams[split_words[0]] + 1
            laplace_bigrams[key] /= (len(words) + len(unique_words) + 1)
            laplace_bigrams[key] += Laplace

            writer.writerow(
                [split_words[0], split_words[1], value,
                MLE, Laplace, Inter, AD]
            )

    # write top-bigrams.txt
    top_bigrams = sorted(laplace_bigrams.items(), key=lambda x:-x[1])[:20]
    with open('top-bigrams.txt', 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter='\t', lineterminator='\n')
        writer.writerow(['word1', 'word2', 'JointLaplace'])
        for item in top_bigrams:
            split_words = item[0].split(' ')
            writer.writerow([split_words[0], split_words[1], item[1]])
