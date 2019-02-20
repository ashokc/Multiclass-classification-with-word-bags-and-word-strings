import numpy as np
import os
import string
import sys
import time
import json
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
import random as rn
#All this for reproducibility
np.random.seed(1)
rn.seed(1)
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk_stopw = stopwords.words('english')

wvLength = 300
vectorSource = str(sys.argv[1]) # none, fasttext, custom-fasttext

def tokenize (text):        #   no punctuation & starts with a letter & between 2-15 characters in length
    tokens = [word.strip(string.punctuation) for word in RegexpTokenizer(r'\b[a-zA-Z][a-zA-Z0-9]{2,14}\b').tokenize(text)]
    return  [f.lower() for f in tokens if f and f.lower() not in nltk_stopw]

def get20News():
    X, labels, labelToName = [], [], {}
    twenty_news = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
    for i, article in enumerate(twenty_news['data']):
        stopped = tokenize (article)
        if (len(stopped) == 0):
            continue
        groupIndex = twenty_news['target'][i]
        X.append(stopped)
        labels.append(groupIndex)
        labelToName[groupIndex] = twenty_news['target_names'][groupIndex]
    nTokens = [len(x) for x in X]
    return X, np.array(labels), labelToName, nTokens

def getEmbeddingMatrix (word_index, vectorSource):
    wordVecSources = {'fasttext' : './vectors/crawl-300d-2M-subword.vec', 'custom-fasttext' : './vectors/' + '20news-fasttext.json' }
    f = open (wordVecSources[vectorSource])
    allWv = {}
    if (vectorSource == 'custom-fasttext'):
        allWv = json.loads(f.read())
    elif (vectorSource == 'fasttext'):
        errorCount = 0
        for line in f:
            values = line.split()
            word = values[0].strip()
            try:
                wv = np.asarray(values[1:], dtype='float32')
                if (len(wv) != wvLength):
                    errorCount = errorCount + 1
                    continue
            except:
                errorCount = errorCount + 1
                continue
            allWv[word] = wv
        print ("# Bad Word Vectors:", errorCount)
    f.close()
    embedding_matrix = np.zeros((len(word_index)+1, wvLength))  # +1 for the masked 0
    for word, i in word_index.items():
        if word in allWv:
            embedding_matrix[i] = allWv[word]
    return embedding_matrix

def sparseMultiply (sparseX, corpus_embedding_matrix):
    denseZ = []
    for row in sparseX:
        newRow = np.zeros(wvLength)
        for nonzeroLocation, value in list(zip(row.indices, row.data)):
            newRow = newRow + value * corpus_embedding_matrix[nonzeroLocation]
        denseZ.append(newRow)
    denseZ = np.array([np.array(xi) for xi in denseZ])
    return denseZ

X, labels, labelToName, nTokens = get20News()
print ('Token Summary. min/avg/median/std/85/86/87/88/89/90/91/92/93/94/95/99/max:',)
print (np.amin(nTokens), np.mean(nTokens),np.median(nTokens),np.std(nTokens),np.percentile(nTokens,85),np.percentile(nTokens,86),np.percentile(nTokens,87),np.percentile(nTokens,88),np.percentile(nTokens,89),np.percentile(nTokens,90),np.percentile(nTokens,91),np.percentile(nTokens,92),np.percentile(nTokens,93),np.percentile(nTokens,94),np.percentile(nTokens,95),np.percentile(nTokens,99),np.amax(nTokens))
labelToNameSortedByLabel = sorted(labelToName.items(), key=lambda kv: kv[0]) # List of tuples sorted by the label number [ (0, ''), (1, ''), .. ]
namesInLabelOrder = [item[1] for item in labelToNameSortedByLabel]
numClasses = len(namesInLabelOrder)
print ('X, labels #classes classes {} {} {} {}'.format(len(X), str(labels.shape), numClasses, namesInLabelOrder))

X=np.array([np.array(xi) for xi in X])          #   rows: Docs. columns: words
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1).fit(X)
word_index = vectorizer.vocabulary_
Xencoded = vectorizer.transform(X)
print ('Vocab sparse-Xencoded {} {}'.format(len(word_index), str(Xencoded.shape)))

if (vectorSource != 'none'):
    embedding_matrix = getEmbeddingMatrix (word_index, vectorSource)
    Xencoded = sparseMultiply (Xencoded, embedding_matrix)
    print ('Dense-Xencoded {}'.format(str(Xencoded.shape)))

# Test & Train Split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1).split(Xencoded, labels)
train_indices, test_indices = next(sss)
train_x, test_x = Xencoded[train_indices], Xencoded[test_indices]
train_labels, test_labels = labels[train_indices], labels[test_indices]
start_time = time.time()
model = LinearSVC(tol=1.0e-6,max_iter=20000,verbose=1)
model.fit(train_x, train_labels)
predicted_labels = model.predict(test_x)
elapsed_time = time.time() - start_time
results = {}
results['confusion_matrix'] = confusion_matrix(test_labels, predicted_labels).tolist()
results['classification_report'] = classification_report(test_labels, predicted_labels, digits=4, target_names=namesInLabelOrder, output_dict=True)

print (confusion_matrix(labels[test_indices], predicted_labels))
print (classification_report(labels[test_indices], predicted_labels, digits=4, target_names=namesInLabelOrder))
print ('Time Taken:', elapsed_time)
results['elapsed_time'] = elapsed_time        # seconds

f = open ('svm-' + vectorSource + '.json','w')
out = json.dumps(results, ensure_ascii=True)
f.write(out)
f.close()

