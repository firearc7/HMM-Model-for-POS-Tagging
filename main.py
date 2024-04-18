import re
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def read_data(training_data):
    with open(training_data, "r", encoding='utf8') as file:
        tagged_data= file.read().split("\n")
    temp= []
    tt= tagged_data
    tagged_data= []
    for i in tt:
        if i=="":
            temp.insert(0, ["<s>", "<s>"])
            temp.append(["</s>", "</s>"])
            tagged_data.append(temp)
            temp= []
        else:
            temp.append(i.split("\t"))
    return tagged_data

def emission_probabilities_table(tagged_training_data):
    emission_probabilities= {}
    for sentence in tagged_training_data:
        for word in sentence:
            if word[1] not in emission_probabilities:
                emission_probabilities[word[1]]= {}
            if word[0] not in emission_probabilities[word[1]]:
                emission_probabilities[word[1]][word[0]]= 1
            else:
                emission_probabilities[word[1]][word[0]]+= 1
    for tag in emission_probabilities:
        total= sum(emission_probabilities[tag].values())
        for word in emission_probabilities[tag]:
            emission_probabilities[tag][word]/= total
    return emission_probabilities

def transition_probabilities_table(tagged_training_data):
    transition_probabilities= {}
    for sentence in tagged_training_data:
        for i in range(len(sentence)-1):
            if sentence[i][1] not in transition_probabilities:
                transition_probabilities[sentence[i][1]]= {}
            if sentence[i+1][1] not in transition_probabilities[sentence[i][1]]:
                transition_probabilities[sentence[i][1]][sentence[i+1][1]]= 1
            else:
                transition_probabilities[sentence[i][1]][sentence[i+1][1]]+= 1
    for tag in transition_probabilities:
        total= sum(transition_probabilities[tag].values())
        for next_tag in transition_probabilities[tag]:
            transition_probabilities[tag][next_tag]/= total
    tags= list(transition_probabilities.keys())
    tags.append("</s>")
    for tag in tags:
        if tag not in transition_probabilities:
            transition_probabilities[tag]= {}
            for next_tag in tags:
                transition_probabilities[tag][next_tag]= 0
    for tag in transition_probabilities:
        for next_tag in tags:
            if next_tag not in transition_probabilities[tag]:
                transition_probabilities[tag][next_tag]= 0
    return transition_probabilities

def viterbi_algorithm(sentence, emission_matrix, transition_matrix):
    words= sentence.split()
    states= list(emission_matrix.keys())
    viterbi= {state: [0 for _ in words] for state in states}
    backpointer= {state: [None for _ in words] for state in states}
    for state in states:
        if words[0] in emission_matrix[state]:
            viterbi[state][0]= transition_matrix['<s>'].get(state, 0) * emission_matrix[state][words[0]]
        else:
            viterbi[state][0]= transition_matrix['<s>'].get(state, 0) * 1e-10
    for t in range(1, len(words)):
        for state in states:
            max_prob, prev_st= max((viterbi[prev_state][t-1] * transition_matrix[prev_state].get(state, 0), prev_state) for prev_state in states)
            viterbi[state][t]= max_prob * emission_matrix[state].get(words[t], 1e-10)
            backpointer[state][t]= prev_st
    max_prob, best_last_state= max((viterbi[state][-1], state) for state in states)
    best_path= [best_last_state]
    for t in range(len(words)-1, 0, -1):
        best_path.insert(0, backpointer[best_path[0]][t])
    return list(zip(words, best_path))

def main():
    # language= input("Enter the language: ")
    # training_data= input("Enter the path of the training data: ")
    # if not os.path.exists(training_data):
    #     if language=="english":
    #         print("File not found, Downloading the training data...")
    #         os.system("wget https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu")
    #         training_data= "en_ewt-ud-train.conllu"
        
    # tagged_training_data= read_data(training_data)
    # emission_matrix= emission_probabilities_table(tagged_training_data)
    # transition_matrix= transition_probabilities_table(tagged_training_data)
    # testing_data= input("Enter a sentence to be tagged: ")

    # predicted_tags= viterbi_algorithm(testing_data, emission_matrix, transition_matrix)


if __name__ == "__main__":
    main()
