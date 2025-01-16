def viterbi_algorithm(sentence, emission_matrix, transition_matrix):
    words= sentence.split()
    states= list(emission_matrix.keys())
    viterbi= {state: [0 for _ in words] for state in states}
    backpointer= {state: [None for _ in words] for state in states}
    for state in states:
        if words[0] in emission_matrix[state]:
            viterbi[state][0]= transition_matrix['<s>'].get(state, 0)*emission_matrix[state][words[0]]
        else:
            viterbi[state][0]= transition_matrix['<s>'].get(state, 0)*1e-10
    for t in range(1, len(words)):
        for state in states:
            max_prob, prev_st= max((viterbi[prev_state][t-1]*transition_matrix[prev_state].get(state, 0), prev_state) for prev_state in states)
            viterbi[state][t]= max_prob*emission_matrix[state].get(words[t], 1e-10)
            backpointer[state][t]= prev_st
    max_prob, best_last_state= max((viterbi[state][-1], state) for state in states)
    best_path= [best_last_state]
    for t in range(len(words)-1, 0, -1):
        best_path.insert(0, backpointer[best_path[0]][t])
    return list(zip(words, best_path))

def beam_search(sentence, emission_matrix, transition_matrix, beam_width=10):
    words= sentence.split()
    states= list(emission_matrix.keys())
    beam= [(1, ['<s>'])]
    for t in range(len(words)):
        new_beam= []
        for score, path in beam:
            for state in states:
                if words[t] in emission_matrix[state]:
                    emission_prob= emission_matrix[state][words[t]]
                else:
                    emission_prob= 1e-10
                for prev_state in path[-1:]:
                    if state in transition_matrix[prev_state]:
                        transition_prob= transition_matrix[prev_state][state]
                    else:
                        transition_prob= 1e-10
                    new_score= score*(transition_prob*emission_prob)
                    new_path= path+[state]
                    new_beam.append((new_score, new_path))
        beam= sorted(new_beam, reverse=True)[:beam_width]
    return list(zip(words, beam[0][1][1:]))

def greedy_search(sentence, emission_matrix, transition_matrix):
    words= sentence.split()
    states= list(emission_matrix.keys())
    path= ['<s>']
    for t in range(len(words)):
        max_prob, best_state= max((emission_matrix[state].get(words[t], 1e-10)*transition_matrix[prev_state].get(state, 0), state) for state in states for prev_state in path[-1:])
        path.append(best_state)
    return list(zip(words, path[1:]))

def posterior_decoding(sentence, emission_matrix, transition_matrix):
    words= sentence.split()
    states= [state for state in emission_matrix.keys() if state not in ['<s>', '</s>']]
    forward= {state: [0 for _ in words] for state in states}
    backward= {state: [0 for _ in words] for state in states}
    forward['<s>']= [1]+[0]*(len(words)-1)
    backward['</s>']= [0]*(len(words)-1)+[1]
    
    for t in range(1, len(words)):
        for state in states:
            forward[state][t]= sum(forward[prev_state][t-1]*transition_matrix[prev_state].get(state, 0)*emission_matrix[state].get(words[t], 1e-10) for prev_state in states+['<s>'])
    
    for t in range(len(words)-2, -1, -1):
        for state in states:
            backward[state][t]= sum(backward[next_state][t+1]*transition_matrix[state].get(next_state, 0)*emission_matrix[next_state].get(words[t+1], 1e-10) for next_state in states+['</s>'])
    
    posterior= {state: [forward[state][t]*backward[state][t] for t in range(len(words))] for state in states}
    return list(zip(words, [max(posterior, key=lambda x: posterior[x][t]) for t in range(len(words))]))
