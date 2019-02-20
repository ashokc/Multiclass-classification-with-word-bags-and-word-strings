import numpy as np
import glob, os, re, sys, json
import matplotlib.pyplot as plt
from PIL import Image

#get_ipython().magic('matplotlib inline')

def plotBars (f1_scores, elapsed_times):
    fig = plt.figure(figsize=(6,6),dpi=720)
    subplot = fig.add_subplot(1, 1, 1)
    width = 0.05
    colors = ['b', 'r', 'g', 'b', 'r', 'g']
    xVals = [0.2, 0.3, 0.4, 0.9, 1.0, 1.1]
    subplot.bar(xVals,f1_scores,width,color=colors)
    subplot.xaxis.set_ticks([])
    subplot.xaxis.set_ticklabels([])
    subplot.set_ylim(bottom=min(f1_scores)-0.01, top=max(f1_scores)+0.01)
    fig.savefig('f1-scores.png', format='png', dpi=720)

    fig = plt.figure(figsize=(6,6),dpi=720)
    subplot = fig.add_subplot(1, 1, 1)
    subplot.bar(xVals,elapsed_times,width,color=colors)
    subplot.xaxis.set_ticks([])
    subplot.xaxis.set_ticklabels([])
    subplot.set_ylim(bottom=min(elapsed_times)-10, top=max(elapsed_times)+10)
    fig.savefig('elapsed-times.png', format='png', dpi=720)

def plotConvergence (results):
    fig = plt.figure(figsize=(6,6),dpi=720)
    subplot = fig.add_subplot(1, 1, 1)
    epochs = list(range(0,len(results['val_acc'])))
    subplot.plot(epochs,results['val_acc'],color='g', label='Validation')
    subplot.plot(epochs,results['val_loss'],color='g')
    subplot.plot(epochs,results['acc'],color='b')
    subplot.plot(epochs,results['loss'],color='b', label='Training')
    subplot.legend(loc='upper right', prop={'size': 10})
    fig.savefig('accuracy.png', format='png', dpi=720)

def main():
    f1_scores, elapsed_times = [], []
    for clf in ['svm', 'lstm']:
        for vectorsource in ['none', 'fasttext', 'custom-fasttext']:
            filename = clf + '-' + vectorsource
            with open (filename+'.json') as fh:
                result = json.loads(fh.read())
                f1_scores.append(result['classification_report']['weighted avg']['f1-score'])
                elapsed_times.append(result['elapsed_time'])
                if ( (clf == 'lstm') and (vectorsource == 'fasttext') ):
                    plotConvergence (result['history'])
    plotBars (f1_scores, elapsed_times)

if __name__ == '__main__':
    main()

