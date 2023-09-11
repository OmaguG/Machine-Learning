
#several import statements to start with
#NLTK - Natural Language Toolkit used for recognition and constrains of human readable text
import nltk
#Allows stem (core) of each word to be pulled out from variation of each word (lenght) to reduce search for the word and improve code performance
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
#numpy - to perform variety of mathematical calculations
import numpy as np
#tflearn is a high-level deep learning library built on the top of TensorFlow
import tflearn
#tensorflow is an end-to-end open-source machine learning platform with a focus on deep neural networks
import tensorflow as tf
#random - for choosing random response in the end
import random
#json - for reading intents saved as .json file
import json
#pickle - for steriliation purposed. This allows partial program code to be run and nor repeated for various input/output combination
import pickle
#reading each word from intents.json as text and loading it into intents dictionary
with open("intents.json") as file:
    data = json.load(file)
#try and except commands are added to loop the training file until all solutions were found (not mandatory but it speeds up the code)
#'rb' stands for read bytes
#without using pickle function, model wil show its calculatons every time and program startup
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
# creating empty lists to store data
    words = []
    labels = []
    docs_x = []
    docs_y = []
# getting into intents dictionary which includes intents as sub-dictionaries
    for intent in data["intents"]:
# looking for pattern within list of patterns for each tag
        for pattern in intent["patterns"]:
# tokenising is taking sentence and splitting it into individual words (not considering symbols)
            wrds = nltk.word_tokenize(pattern)
#all of the words returned by tokenising will be saved into the words list
            words.extend(wrds)
#each entry in decs_x will now correspond with entry in docs_y (important for training the model
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
#this part of code searcheds through all the tags and if tag not found in tags, it will be added
#this function is added to assure if selected word is not in tag category, to add it to the category
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
#all of the words will be converted to lower case, and stem (core) of each word will be pulled out to simplify model and reduce number of words
# specifying possible symbols in intents.json file to be ignored for training the algorythm
    words = [stemmer.stem(w.lower()) for w in words if w != ["?",".",",","!","'"]]
# in case of duplicates in the list of words and labels, they will be removed with set command
    words = sorted(list(set(words)))
    labels = sorted(labels)

####### CREATING TRAINING AND TESTING INPUT #######

# All of the words,classes, characters previously saved into the lists or classes need to be converted into numerical values.
# neural network need numerical values for learning

# opening empty training list to save all training data

    training = []
    output = []
# template of zeroes for the lenght of each word for training data
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]
# for each of the combinations there will be and empty bag of words (bunch of zeroes and ones coresponding to all possible words)
# going through previously saved list of words and assigning it to 1 if word exists and 0 if it doesn't (One HOT encoding)
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

                ####### CREATING TRAINING AND TESTING OUTPUT #######

# [:] is only a short version of previously typed in [0 for _ in range(len(labels))]
        output_row = out_empty[:]
# searching through the labels list and assigning to 1 if word exists in the list
        output_row[labels.index(docs_y[x])] = 1
# pulled out info from training and output lists will now be attached into the lists
# adding the result above into the training list previously created
        training.append(bag)
        output.append(output_row)
#both lists are now converted into numpy arrays in structured way of x and y
# this step is to create x-features and y-labels to train the neural network
    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

##### NEURAL NETWORK MODEL BUILDING #####

#resetting the underlying data graph (previously stored settings)

tf.compat.v1.reset_default_graph()
#the shape of the model would be related to number of words in training data and depending of its lenght
net = tflearn.input_data(shape=[None, len(training[0])])
#addition of two hidden layers with 8 neurons each fully conected to input data (two hiden layers are enough for this simple model)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
#adding another layer with as many neurons as number of classes and sorting them by % from lowest to highest
#this allows probabilities for each neuron (output) in each layer
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# specifying the learning rate and parameters of the gradient descent as build in function. SIMILAR to separate SGD parameter
net = tflearn.regression(net)
# DNN is one of many types or neural networks that could have been potentially used
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    ##### FITTING THE MODEL #####

# start passing the training data
# specifying training data repetition for learing (each data fed 1000 times as default for training) and size of batch for each training record
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
# saving already created model
    model.save("model.tflearn")

#turning sentence input from the user into a bag of words
def bag_of_words(s, words):
        bag = [0 for _ in range(len(words))]
# separating sentence into single words
        s_words = nltk.word_tokenize(s)
# taking stem of the word from each word
        s_words = [stemmer.stem(word.lower()) for word in s_words]
# looking for current word in the words list and its equivalent on the sentence
        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
# append this word as a value of 1 if that word exists
                    bag[i] = 1
# load results into numpy array for structure
        return np.array(bag)

#Adding a change to style and colour of the font

import colorama
colorama.init()
from colorama import Fore, Style
##### CHAT INTERFACE #####
def chat():
        global responses
# welcome message to start the chat
        print(Fore.LIGHTGREEN_EX + "WELCOME - Start messaging with AdiBot chatbot (type quit to stop or human if you wish to speak to advisor)!" + Style.RESET_ALL)
        while True:
            inp = input("You: ")
# in case of user wants to finish chat, it is required to type'quit'
            if inp.lower() == "quit":
                break
# in case of user wants to finish bit bot and connect to human type'human'
            if inp.lower() == "human":
                print("Click on the link to connect via live chat https://www.adidas.co.uk/help")
                break
# model.predict makes predicition on the multiple things at the time from bunch of inputs and possible outputs
# if code was finished at this stage it will only show numbers = probabilities of each answer in numerical state
            results = model.predict([bag_of_words(inp, words)])[0]
# argmax gives index of the greatest value from the list
            results_index = np.argmax(results)
# this allows the numerical value to be displayed as real tag value from iintents file
            tag = labels[results_index]
# now it is important to get into intents file, and select only possibilities which match 70% of the possible answers and display them

            if results[results_index] > 0.7:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                print(Fore.LIGHTBLUE_EX + "Chatbot:" + Style.RESET_ALL, random.choice(responses))
# if user input is not understood by model (input words not matching training words) then bot will ask for another input
            else:
                print("Sorry I did't get that, could you please rephrase the question?")

chat()
