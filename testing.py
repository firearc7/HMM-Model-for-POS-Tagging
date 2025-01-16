from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
from algorithms import *

def test_viterbi(tagged_testing_data, emission_matrix, transition_matrix):
    actual, predicted= [], []
    for sentence in tagged_testing_data:
        sentence= sentence[1:-1]
        s= " ".join([word[0] for word in sentence])
        actual_tags= [word[1] for word in sentence]
        predicted_tags= viterbi_algorithm(s, emission_matrix, transition_matrix)
        for i in range(len(actual_tags)):
            try:
                predicted.extend([predicted_tags[i][1]])
                actual.extend([actual_tags[i]])
            except:
                pass
    return actual, predicted

def test_beam_search(tagged_testing_data, emission_matrix, transition_matrix, beam_width=3):
    actual, predicted = [], []
    for sentence in tagged_testing_data:
        sentence = sentence[1:-1]
        s = " ".join([word[0] for word in sentence])
        actual_tags = [word[1] for word in sentence]
        predicted_tags = beam_search(s, emission_matrix, transition_matrix, beam_width)
        for i in range(len(actual_tags)):
            try:
                predicted.extend([predicted_tags[i][1]])
                actual.extend([actual_tags[i]])
            except:
                pass
    return actual, predicted

def test_greedy_search(tagged_testing_data, emission_matrix, transition_matrix):
    actual, predicted= [], []
    for sentence in tagged_testing_data:
        sentence= sentence[1:-1]
        s= " ".join([word[0] for word in sentence])
        actual_tags= [word[1] for word in sentence]
        predicted_tags= greedy_search(s, emission_matrix, transition_matrix)
        for i in range(len(actual_tags)):
            try:
                predicted.extend([predicted_tags[i][1]])
                actual.extend([actual_tags[i]])
            except:
                pass
    return actual, predicted

def test_posterior_decoding(tagged_testing_data, emission_matrix, transition_matrix):
    actual, predicted= [], []
    for sentence in tagged_testing_data:
        sentence= sentence[1:-1]
        s= " ".join([word[0] for word in sentence])
        actual_tags= [word[1] for word in sentence]
        predicted_tags= posterior_decoding(s, emission_matrix, transition_matrix)
        for i in range(len(actual_tags)):
            try:
                predicted.extend([predicted_tags[i][1]])
                actual.extend([actual_tags[i]])
            except:
                pass
    return actual, predicted

def confusion_matrix(actual, predicted):
    tags= list(set(actual))
    tags.sort()
    c= pd.DataFrame(np.zeros((len(tags), len(tags))), index=tags, columns=tags)
    for i in range(len(actual)):
        if actual[i] in c.index and predicted[i] in c.columns:
            c.loc[actual[i], predicted[i]]+= 1
    return c

def evaluation(actual, predicted):
    precision= precision_score(actual, predicted, average='weighted', zero_division=0)
    recall= recall_score(actual, predicted, average='weighted', zero_division=0)
    f1= f1_score(actual, predicted, average='weighted')
    c= confusion_matrix(actual, predicted)
    return precision, recall, f1, c
