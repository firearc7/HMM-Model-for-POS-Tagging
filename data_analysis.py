import os

def read_file_from_web(url, filename):
    import requests
    r= requests.get(url)
    r= r.content.decode("utf-8")
    r= r.split("\n")
    i= 0
    while i<len(r):
        if r[i].startswith("#"):
            r.pop(i)
        elif r[i]!="":
            r[i]= r[i].split("\t")
            r[i]= r[i][1]+"\t"+r[i][3]
            i+= 1
        elif r[i]=="":
            i+= 1
    r= "\n".join(r)
    if r.endswith("\n\n"):
        r= r[:-1]
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(r.encode("utf-8"))

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
