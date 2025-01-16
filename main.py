import os
import time
from data_analysis import *
from algorithms import *
from testing import *

def main():
    language= input("Enter the language: ")
    language= language.lower()
    training_data= input("Enter the data: ")
    if not os.path.exists(training_data):
        if language=="english":
            print("File not found, Downloading the training data from the web...")
            read_file_from_web("https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu", "Dataset/english.txt")
            training_data= "Dataset/english.txt"
        elif language=="hindi":
            print("File not found, Downloading the training data from the web...")
            read_file_from_web("https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/master/hi_hdtb-ud-train.conllu", "Dataset/hindi.txt")
            training_data= "Dataset/hindi.txt"
        elif language=="spanish":
            print("File not found, Downloading the training data from the web...")
            read_file_from_web("https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master/es_gsd-ud-train.conllu", "Dataset/spanish.txt")
            training_data= "Dataset/spanish.txt"
        elif language=="sanskrit":
            print("File not found, Downloading the training data from the web...")
            read_file_from_web("https://raw.githubusercontent.com/UniversalDependencies/UD_Sanskrit-Vedic/master/sa_vedic-ud-train.conllu", "Dataset/sanskrit.txt")
            training_data= "Dataset/sanskrit.txt"
        else:
            print("File not found, no pre-existing dataset for this language, cannot train the model. Exiting...")
            exit()
    print("Reading the training data...")
    tagged_training_data= read_data(training_data)
    tagged_testing_data= tagged_training_data[:int(0.1*len(tagged_training_data))]
    tagged_training_data= tagged_training_data[int(0.1*len(tagged_training_data)):]
    print("Training the model...")
    emission_matrix= emission_probabilities_table(tagged_training_data)
    transition_matrix= transition_probabilities_table(tagged_training_data)
    print("Training Completed!")
    print("Testing Viterbi Algorithm...")
    start_time= time.time()
    actual_viterbi, predicted_viterbi= test_viterbi(tagged_testing_data, emission_matrix, transition_matrix)
    precision_viterbi, recall_viterbi, f1_viterbi, c_viterbi= evaluation(actual_viterbi, predicted_viterbi)
    filename="Results/"+language+"/viterbi_results.txt"
    with open(filename, "w") as f:
        f.write("Precision: "+str(precision_viterbi)+"\n")
        f.write("Recall: "+str(recall_viterbi)+"\n")
        f.write("F1 Score: "+str(f1_viterbi)+"\n")
        f.write("Confusion Matrix:\n"+str(c_viterbi)+"\n")
    end_time= time.time()
    print("Viterbi Results saved in Results folder!")
    print("Time taken for Viterbi Algorithm: ", end_time-start_time, " seconds")
    print("Testing Beam Search Algorithm...")
    start_time= time.time()
    actual_beam_search, predicted_beam_search= test_beam_search(tagged_testing_data, emission_matrix, transition_matrix)
    precision_beam_search, recall_beam_search, f1_beam_search, c_beam_search= evaluation(actual_beam_search, predicted_beam_search)
    filename= "Results/"+language+"/beam_search_results.txt"
    with open(filename, "w") as f:
        f.write("Precision: "+str(precision_beam_search)+"\n")
        f.write("Recall: "+str(recall_beam_search)+"\n")
        f.write("F1 Score: "+str(f1_beam_search)+"\n")
        f.write("Confusion Matrix:\n"+str(c_beam_search)+"\n")
    end_time= time.time()
    print("Beam Search Results saved in Results folder!")
    print("Time taken for Beam Search Algorithm: ", end_time-start_time, " seconds")
    print("Testing Greedy Search Algorithm...")
    start_time= time.time()
    actual_greedy_search, predicted_greedy_search= test_greedy_search(tagged_testing_data, emission_matrix, transition_matrix)
    precision_greedy_search, recall_greedy_search, f1_greedy_search, c_greedy_search= evaluation(actual_greedy_search, predicted_greedy_search)
    filename= "Results/"+language+"/greedy_search_results.txt"
    with open(filename, "w") as f:
        f.write("Precision: "+str(precision_greedy_search)+"\n")
        f.write("Recall: "+str(recall_greedy_search)+"\n")
        f.write("F1 Score: "+str(f1_greedy_search)+"\n")
        f.write("Confusion Matrix:\n"+str(c_greedy_search)+"\n")
    end_time= time.time()
    print("Greedy Search Results saved in Results folder!")
    print("Time taken for Greedy Search Algorithm: ", end_time-start_time, " seconds")
    print("Testing Posterior Decoding Algorithm...")
    start_time= time.time()
    actual_posterior_decoding, predicted_posterior_decoding= test_posterior_decoding(tagged_testing_data, emission_matrix, transition_matrix)
    precision_posterior_decoding, recall_posterior_decoding, f1_posterior_decoding, c_posterior_decoding= evaluation(actual_posterior_decoding, predicted_posterior_decoding)
    filename= "Results/"+language+"/posterior_decoding_results.txt"
    with open(filename, "w") as f:
        f.write("Precision: "+str(precision_posterior_decoding)+"\n")
        f.write("Recall: "+str(recall_posterior_decoding)+"\n")
        f.write("F1 Score: "+str(f1_posterior_decoding)+"\n")
        f.write("Confusion Matrix:\n"+str(c_posterior_decoding)+"\n")
    end_time= time.time()
    print("Posterior Decoding Results saved in Results folder!")
    print("Time taken for Posterior Decoding Algorithm: ", end_time-start_time, " seconds")

if __name__ == "__main__":
    main()
