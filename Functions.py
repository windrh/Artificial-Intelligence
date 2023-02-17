import random
import copy
import math

#function takes a number and returns the result of the number passed through the sigmoid function
def sigmoid(y):
    try:
        maths = 1 / (1 + (math.e ** -y))
    except:
        maths = 0
    return maths

#function takes a value and returns the ReLU output of that value.
def LeakyReLU(x):
    if x < 0:
        return x * .1
    else:
        return x

#function takes a value (assuming it is a value derived from the ReLU function) and gives the derivative that value.
def dLeakyReLU(x):
    if x < 0:
        return -.1
    else:
        return 1

#function takes a value (assuming it is a value derived from the Sigmoid function) and gives the derivative of that vlaue.
def dsigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))


#gives a random weight initialization according to the Xiaver Weight Distribution Formula
def weight_initialization(number_of_input_nodes):
    x = -(1 / math.sqrt(number_of_input_nodes))
    y = (1 / math.sqrt(number_of_input_nodes))
    return random.uniform(x, y)


#compound function that generates a formatted list of randomly initlized weights.
def initialize_starting_weights(number_inputs, length_of_hidden_layers, num_output_nodes, number_of_hidden_layers=1):
    finallist = []
    temptable = []
    temporarylist = []
    for i in range(number_inputs):
        for x in range(length_of_hidden_layers):
            temporarylist.append(weight_initialization(number_inputs))
        temptable.append(copy.deepcopy(temporarylist))
        temporarylist.clear()
    finallist.append(copy.deepcopy(temptable))
    temptable.clear()
    for i in range(number_of_hidden_layers-1):
        for x in range(length_of_hidden_layers):
            for y in range(length_of_hidden_layers):
                temporarylist.append(weight_initialization(number_inputs))
            temptable.append(copy.deepcopy(temporarylist))
            temporarylist.clear()
        finallist.append(copy.deepcopy(temptable))
        temptable.clear()
    for i in range(length_of_hidden_layers):
        for x in range(num_output_nodes):
            temporarylist.append(weight_initialization(number_inputs))
        temptable.append(copy.deepcopy(temporarylist))
        temporarylist.clear()
    finallist.append(copy.deepcopy(temptable))
    temptable.clear()
    return finallist

#compund function that randomly generates a list of biases.
def initialize_starting_biases(number_inputs,number_outputs,length_hidden_layers, number_of_hidden_layers=1):
    biastablelist = []
    templist = []
    for x in range(number_of_hidden_layers+1):
        if x < number_of_hidden_layers:
            for i in range(length_hidden_layers):
                templist.append(weight_initialization(number_inputs))
            biastablelist.append(copy.deepcopy(templist))
            templist.clear()
        else:
            for i in range(number_outputs):
                templist.append(weight_initialization(number_inputs))
            biastablelist.append(copy.deepcopy(templist))
            templist.clear()
    return biastablelist

#reads a .txt file and returns the model in list format.
def read_model(filename):
    a = open(filename, 'r')
    b = a.read()
    return eval(b)

#stores a model into any given filename
def store_model(filename,model):
    a = open(filename, 'w')
    a.write(str(model))
    a.close()

#this function returns 0 if there is no file given, and if there is a file given it returns a filename.
def inquire_data(answer = None):
    if answer is None:
        return 0
    else:
        return answer

#This function takes an input which tell it to take in parameters to generate a new model or read a pre-exisitng model.
def produce_data(number):
    if number == 0:
        print("Creating new model for the neural network... \n")
        print("Input the following values in respect to the order given below:")
        print("number of inputs,length of hidden layers,number of outputs,number of hidden layers")
        parameters = input("Paramters: ")
        newparam = parameters.split(",")
        print("\nInput order of activation functions in a single string, " + str(int(newparam[3])+1) + " long, of numbers according to the following guide:")
        print("0 for LeakyRelu")
        print("1 for Sigmoid")
        finallist = []
        finallist.append(initialize_starting_biases(int(newparam[0]),int(newparam[2]),int(newparam[1]),int(newparam[3])))
        finallist.append(initialize_starting_weights(int(newparam[0]),int(newparam[1]),int(newparam[2]),int(newparam[3])))
        activefunct = input("Activation function string: ")
        templist = []
        for elem in activefunct:
            templist.append(int(elem))
        finallist.append(templist)
        print("\nModel initialized successfully")
        return finallist
    else:
        print("Model read successfully")
        return read_model(number)

#GROUP FUNCTIONS
# three functions index values inside of the model's nesting, allowing them to be represented as a matrix
def bias_coord(variable_name_of_model,layer_number,specific_node):
    return variable_name_of_model[0][layer_number][specific_node]

def weight_coord(variable_name_of_model,matrix_number,row,column):
    return variable_name_of_model[1][matrix_number][row][column]

def activation_funct_coord(variable_name_of_model,layer):
    return variable_name_of_model[2][layer]

#forward propogates data through the model with a computer iteration of matrix dot multiplication
def forward_prop(name_of_model_variable,input_data):

    activated_nodes_list = []
    currentnode = []

    #puts the input data in the current node list so that it can be used recursively for purposes in the first iteration.
    for element in input_data:
        currentnode.append(element)


    #determines which layer the model is on
    for x in range(len(name_of_model_variable[2])):
        temporary_activated_node_list = []
        temporary_node_list = []

        #creates a value of 0 for each future node in the temporary node list
        for tmp in range(len(name_of_model_variable[0][x])):
            temporary_node_list.append(0)

        #multiply the nodes by the matrix weights for respective layer
        for y in range(len(name_of_model_variable[1][x])): #for every row in a specific matrix
            for z in range(len(name_of_model_variable[1][x][y])): #for every column value in a specific matrix
                temporary_node_list[z] += weight_coord(name_of_model_variable,x,y,z) * currentnode[y]

        #add the bias to each new node
        for b in range(len(name_of_model_variable[0][x])):
            temporary_node_list[b] += bias_coord(name_of_model_variable,x,b)

        #pass each new node value through an activation function
        if name_of_model_variable[2][x] == 0:
            for elemx in temporary_node_list:
                temporary_activated_node_list.append(LeakyReLU(elemx))
        elif name_of_model_variable[2][x] == 1:
            for elemy in temporary_node_list:
                temporary_activated_node_list.append(sigmoid(elemy))

        #clears out existing variables and updates relevant data structures for the continuation of the sequence
        activated_nodes_list.append(temporary_activated_node_list.copy())
        currentnode.clear()
        for node in temporary_activated_node_list:
            currentnode.append(node)
        temporary_activated_node_list.clear()

    return activated_nodes_list


#backpropogates a model given the variable name of the A.I. model, a desired learning rate, how many iterations you want to backprop, initial input data, and correct output data.
    #returns nothing, however it updates the A.I. model that it is given within local computer storage. MUST STORE THE NEW MODEL SOMEWHERE FOR IT TO BE ACCESSIBLE AGAIN
def backprop(models_variable_name,learning_rate,iterations,inputdata,correct_output_data):
    #outcome of forward propagation for error calculations
    inputresult = forward_prop(models_variable_name,inputdata)

    #actual class definition for the nodes we use to find derivatives dynamically
    class Node(object):
        def __init__(self,activated_value,activation_function,incominglist,outgoinglist,name_of_list,x,y,end = False):
            self.activated_value = activated_value #INT | actual output of the node
            self.activation_function = activation_function #INT | abstract values of 0 or 1 that correspond to LeakyRelu or Sigmoid
            self.incominglist = incominglist #LIST | each list inside this list are coordinates inside to weights within the given model
            self.outgoinglist = outgoinglist #LIST | each list inside this list are coordinates inside to weights within the given model
            self.x = x #INT | abstract value for human purposes of creating the nodes
            self.y = y #INT | abstract value for human purposes of creating the nodes
            self.end = end #BOOL | a simple boolean value to mark whether this node is at the end of the model, useful for determining whether to send out data to child nodes or not
            self.storage = [] #LIST | acts as a temporary place to hold incoming data and to send it out, can be cleared because intrinsic data is stored in other object attributes
            self.deactivatedvalue = self.valuefinder(self.activation_function, self.activated_value) #Derivative of the node's final output for gradient calculations
            self.name_of_list = name_of_list #VARIABLE | this variable represents the list that this class is being stored into

        #simple function to calculate the derivative of the activated value needed for the gradient
        def valuefinder(self,act_funct,act_value):
            if act_funct == 0:
                return dLeakyReLU(act_value)
            elif act_funct == 1:
                return dsigmoid(act_value)
            else:
                print("error in valuefinder function")

        #class method to turn self.end true from the outside
        def turnonend(self):
            self.end = True

        #part of a semi-recursive structure for finding the derivatives through the network for an A.I.
        def sendout(self):
            for element in self.name_of_list: #searches the all nodes
                for item in element.incominglist: #searches that specific node's incoming list
                    for item2 in self.outgoinglist: #searches its own outgoing list
                        if item == item2: #match of items = there is a 'connection' for data values to be sent
                            templist = [] #creates a temporary list for sending out values
                            for value in self.storage: #appends currently stored values in the list
                                templist.append(value)
                            templist.append(self.deactivatedvalue) #appends its own data, this is the derivative of activation function
                            templist.append(item) #appends its own data part 2, this data is the specific WEIGHT coordinate for backprop
                            element.accept(templist) #calls on the matching node's "accept" function to transfer data
                            templist.clear() #clears out list to reduce memory usage
            self.storage.clear() # clears out the storage so that future items can be stored in later derivative calculations

        #this function call determines conditionally whether to pursue sending out values to continue the chain of derivatives
        def propagate(self):
            if self.end == False:
                self.sendout()

        #accepts a list of coordinates and deactivated values into self.storage for future functions
        def accept(self,listinput):
            for element in listinput:
                self.storage.append(element)
            self.propagate()

    #creating a list of node values so that we can use them as objects
    #need to make a quick for and if loop to turn on the end boolean for nodes using max x value
    node_list = []
    for x in inputresult:
        for y in x:
