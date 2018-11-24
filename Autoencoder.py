#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 13:31:12 2018

@author: eva
"""
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt 
import pickle

    
######################## LOSSES AND ACTIV. FUNCTIONS  #####
def MSELoss(out, truth):
    """MSE Loss, inputs are np arrays."""
    return np.sum((out - truth)**2) / truth.size

def MSELossPrime(out,truth):
    """Derivative of MSE Loss, inputs are np arrays."""
    return 2*(out-truth)/truth.size



def CategoricalCrossEntropy(out, truth):
    """CCE Loss, inputs are np arrays."""
    return 0

def CategoricalCrossEntropyPrime(out, truth):
    """CCE Loss derivative, inputs are np arrays."""
    return 0




def relu(thing):
    return np.maximum(thing, 0)

def reluPrime(thing):
    return thing>0

def sigmoid(thing):
    return 1/(1+np.exp(-thing))

def sigmoidPrime(thing):
    return sigmoid(thing)*(1-sigmoid(thing))

def softmax():
    return 0

def softmaxPrime():
    return 0


###########################################################


######################## FORWARD & BACKWARD PASSING  ######
    
def forwardPassToNextLayer(inVec, weights):
    """ inVec is one image on which to do the forward pass. 
        (Size 28x28, but flattened, for MNIST). Weights 
        is a matrix of size encoding_size x 784(+1 for bias). 
        Returns nxt layer featuremap."""
    #flatten the image before inputing i, and add 1 for bias!
    #img = img.flatten()
    inVec = np.append(inVec, [[1]], axis=0)
    encoded = np.dot(weights, inVec)
    return encoded

#def forwardPassToDecoded(encoding, weights):
#     """encoding is encoded image from which to do the forward pass. 
#        Weights is a matrix of size decoded_size x encoding_size
#        Returns decoded image."""
#     decoded = np.dot(weights, encoding)
#     return decoded


def initializeWeights(weightsfrom, weightsto, method = 'uniform'):
    """randomly initializes weights, returns them
       in a matrix of size weightsfrom x weightsto.
       Type of initialization is decided by method 
       parameter: uniform, Xavier"""
    if method == "uniform":
        return np.random.uniform(size = (weightsto, weightsfrom))
    elif method == "Xavier":
        return np.random.normal(0,1,size=(weightsto, weightsfrom))/np.sqrt(weightsto+weightsfrom)
    raise NameError('Method not recognized')
           

#def update_weights(w1, w2, w1_up, w2_up, learningRate):
#    w1_up, w2_up = w1_up/np.maximum(np.max(w1_up), 1), w2_up/np.maximum(np.max(w2_up), 1)
#    w1 = w1-w1_up*learningRate
#    w2 = w2-w2_up*learningRate
#    return w1, w2
       
def update_weights(w, w_up, learningRate):
    """updates weights given in the tuple w, with 
       corresponding, equally sized, w_up updates, 
       using a learning rate learningRate...GD."""
    new = list()
    for wi, wi_up in zip(w, w_up):
        #wi_up = wi_up/np.maximum(np.max(wi_up), 1)
        wi = wi-wi_up*learningRate
        #instead of normal. update, normalize final weights.
        wi = (wi - np.mean(wi))/np.std(wi)
        new.append(wi)
    return tuple(new)


#def backpropagate(img, out, encoded, weights2):
#    """backprop for special case, sigmoid/relu+MSE, 1 hidden layer only.
#    Returns updates for weights."""
#    dLdY = MSELossPrime(out, img).reshape(-1, 1) #make it a column vector
#    #weights2_update = np.dot(dLdY*sigmoidPrime(out),encoded.reshape(1, -1))
#    #weights1_update = np.dot(np.dot(weights2.T, dLdY*sigmoidPrime(out)), (img*sigmoidPrime(out)).reshape(1,-1))
#    weights2_update = np.dot(dLdY, encoded.reshape(1, -1))
#    weights1_update = np.dot(np.dot(weights2.T, dLdY), img.reshape(1,-1))
#    return (weights1_update, weights2_update)

def backpropagate(truth, out, forwardPasses, weights, lossPrime=MSELossPrime, activationPrime=reluPrime, activation=relu):
    """backprop. Calculates derivatives needed for updates.
       truth = GT, out = final result of the forward pass
       forwardPasses = intermediate results from forward 
       passing. Includes also the last one (i.e. 'out' but 
       without the nonlinearity. weights = tuple of all weights.
       Uses results from forward passes (neuron outputs 
       before application of activation funcs)."""
    dLdY = lossPrime(out, truth).reshape(-1, 1) #make it a column vector
    
    #at every forward pass, we save neuron outputs before nonlinearity.
    #so forwardPasses will tell us how many layers we have:
    forsta = dLdY * activationPrime(forwardPasses[-1]).reshape(-1,1) #now a column vector, n x 1
    bias_updates = [forsta]
    weight_updates = [np.dot(forsta, activation(forwardPasses[-2]).T)] 
    
    for layer in range(len(forwardPasses)-1, 0,-1) :
        #here we need to multiply by weight matrices, but without the bias terms... 
        #so omit last column when multiplying.
        tmp_weights = weights[layer].T
       
        tmp = np.dot(tmp_weights[0:-1,], bias_updates[0]) * activationPrime(forwardPasses[layer-1])
        bias_updates = [tmp] + bias_updates #bias_updates.insert(0, tmp)
        weight_updates = [np.dot(tmp, activation(forwardPasses[layer-2]).T)] + weight_updates #weight_updates.insert(0, np.dot(tmp, activation(forwardPasses[layer-2]).T))
    
    #now weight_updates and bias_updates are tuples that need to be joined for proper update. 
    updates=[]
    for bias, weight in zip(bias_updates, weight_updates):
        updates = updates + [(np.concatenate((weight, bias), axis=-1))]
    return updates
    
    
    
######################   M A I N S  ####################


#encoded_size = 350
#img_size = 28*28 + 1 
#epochs = 250
#learningRate = 0.2


def training(X_train, Y_train, list_of_layer_sizes, epochs, LR, LRdecay=True, activationFun='relu', lastActivation='relu', Loss='MSE', initial='Xavier', save=True, verbose=True):
    """main function for the training. Does forward pass and backprop.
       Does batch training if desired (TODO). Saves weights every epoch
       and prints losses. 
       Parameter list: X_ an Y_train is training data, list_of_layer_sizes
       contains sizes of ALL layers including input and output. ...
           """
           
           #TODO!!! add possibility of inputting weights, so you cand do additional trainng on pretrained weights!
    #parameter setting
    actAndLossDict = {'relu':(relu, reluPrime), 'sigmoid':(sigmoid, sigmoidPrime), 'softmax':(softmax, softmaxPrime), 'MSE': (MSELoss, MSELossPrime), 'cce':(CategoricalCrossEntropy, CategoricalCrossEntropyPrime)}
    (activationFun, activationFunPrime) = actAndLossDict[activationFun]
    (lastActivation, lastActivationPrime) = actAndLossDict[lastActivation]
    (lossFun, lossPrime) = actAndLossDict[Loss]
    epoch_losses = np.zeros(epochs)
        
    #initialize weights:
    N = len(list_of_layer_sizes) #including first=in and last=out layer
    weights = list()
    for i in range(1,N):
        weights.append(initializeWeights(list_of_layer_sizes[i-1] + 1, list_of_layer_sizes[i], method = initial)) #+1 here is for bias
    weights = tuple(weights)
   
    
    #setup for batch learning
    L = int(len(X_train)/30)
    batch_size = 30 ## change this if you want batches
    batches = L #==len(X_train),  change this if you want batches
    #
    for ep in range(epochs):
        loss = 0
        if verbose:
            print("Epoch ", (ep+1), "...")

        #setup for batch learning:
        learning_in_batches = np.random.permutation(L) #make a random permutation, to have a random distribution into batches at every epoch.
        X_train = X_train[learning_in_batches]
        #
        for btch in range(batches):
            #setup for batch learning, wont do anything if learning per image
            x_batchtrain = X_train[batch_size*btch : min(batch_size*btch + batch_size, L)] #without batch learning, this will give only 1 image every time
            
            batch_updates = [0 for i in range(N-1)] 
            for im in x_batchtrain: #now go through all the images in this batch... 
                #first propagate forward, save intermediate inputs (before activ. function):
                origInput = (im.flatten()/255).reshape(-1,1)
                intermediateOutputs = [forwardPassToNextLayer(origInput, weights[0])] #doesn't save original input, but all others layers yes.
                for l in range(N-2):
                    intermediateOutputs.append(forwardPassToNextLayer(activationFun(intermediateOutputs[l]), weights[l+1]))
        
                #now that image is forward propagated and intermediate results are saved, we can back propagate
                finalOut = lastActivation(intermediateOutputs[-1]).reshape(-1,1)
                updates = backpropagate(origInput, finalOut, intermediateOutputs, weights, lossPrime=lossPrime, activationPrime=activationFunPrime, activation=activationFun)
                loss = loss + lossFun(finalOut, origInput)
                #for proper batch learning: add backprop/update based on average loss or sth?  Herregud
                #the backprop in that case probably needs to be put outside the batch processing... skiten
                #TODO - I think it should be like this:
                upd = 0
                for i in zip(batch_updates, updates):
                    batch_updates[upd] = sum(i)
                    upd = upd+1
            #now se avg updates for each weight?
            updates = [ bbuu/batch_size for bbuu in batch_updates] ## TODO: you're not supposed to justdivide by batch size,sinze last batch might be smaller
            #how does batch learning really work för fan?!
            #print(lossFun(finalOut, origInput))
    
            #now backprop step is done, update the weights for this batch :)
            if LRdecay and (ep+1)%10==0:
                LR = LR*0.5
            weights = update_weights(weights, updates, LR)
            #print("weights updated for: (", np.min([np.min(updt) for updt in updates]), ", ", np.max([np.max(updt) for updt in updates]), ") \n" )
        
        epoch_losses[ep] = (loss/len(X_train))
        #save results from this epoch
        if save:
            f = open('weights{0}.pckl'.format(ep), 'wb')
            pickle.dump(weights, f)
            f.close()
            
        if verbose:
            print("Avg loss for this epoch: ", (loss/L), "\n")
            
    return weights
            



def testa(x_test, y_test, weights, lossFunk, activationFun, lastActivation, verbose=True): ##TODO: add possibility of different last activ.fun
    """main function for testing, that does forward prop
       of the input images based on the weights tuple.
       Outputs results and (prints) errors based on lossFun."""
       
    # TODO: style does not comply with other funct: here you need to input actual
    # activ, funct, not just the string name. So change this here or make additional wrapper...!!!
    N = len(weights)   
    results=[]
    losses=[]
    for sl in x_test:
        origInput = (sl.flatten()/255).reshape(-1,1)
        result = forwardPassToNextLayer(origInput, weights[0])
        for l in range(N-1):
            result = forwardPassToNextLayer(activationFun(result), weights[l+1])
        result = lastActivation(result)
        results.append(result)
        loss = lossFunk(result, origInput)
        losses.append(loss)
    return (results, losses)
               
        

def plotExamples(trueOut, myOut, loss, howmany, showloss=True, save=False):
    """For plotting some example results together with corresponding ground
       truths. showLoss toggles visibility of per example loss. howmany
       tells the number of examples we want to visualize (could be omitted, 
       but TODO check special case of visualizing 1 datapoint!
       Funciton assumes that myOut is flattened and reshapes it to size of
       trueOut. Also un'normalizes by grayscale factor 255.
       Returns figure holder, saves figure if option save set to true."""
    fig = plt.figure()
    cols = int(np.sqrt(howmany))
    rows = (howmany)//cols + ((howmany)%cols!=0)
    cols,rows = cols*2,rows*2
    
    fig.set_size_inches(rows*2, cols*1.5)
    for i in range(0, howmany):
        plt.subplot(rows,cols, 2*i+1)
        plt.imshow(trueOut[i])
   
        plt.subplot(rows,cols, 2*i+2)
        dec = myOut[i]
        plt.imshow(dec.reshape(trueOut[0].shape)*255)
        if showloss:
            plt.xlabel(round(loss[i],5))
            
    if save:
        fig.savefig('rezultati.png')
        plt.close()
    return fig


################ Calling stuff. 
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#training(x_train[1:50], y_train[1:50], [784,250,100,250,784], 5, 0.1)






finalWeights=training(a_train, b_train, [784,250,100,250,784], 30, 1)











######################################################33
 
#f = open('weights29.pckl', 'rb')
#weights = pickle.load(f)
#f.close()

######   OLD CODE


#set up weights.
#w1 = initializeWeights(img_size, encoded_size)
#w2 = initializeWeights(encoded_size, img_size)

#avg_losses = np.zeros(epochs)
#N = 20
#
#for ep in range(epochs):
#    if loss > avg_losses[ep-2]: #ep%15==0:
#        learningRate = learningRate/2
#    
#    if abs(loss - avg_losses[ep-2])<0.00001 and loss>0.1:
#        learningRate = 0.25
#    loss = 0
#    
#    for i in range(N): #y_train.size):
#        #forward pas, get loss, propagate bkw
#        im = np.append(x_train[i].flatten()/255, 1).reshape(-1,1)
#        
#        enc = forwardPassToEncoded(im, w1)
#        enc = relu(enc)
#        #enc = sigmoid(enc)
#        dec = forwardPassToDecoded(enc, w2)
#        dec = relu(dec)
#        #dec = sigmoid(dec)
#        loss = loss + MSELoss(dec, im)
#        
#        (w1_up, w2_up) = backpropagate(im, dec, enc, w2)
#        w1, w2 = update_weights(w1, w2, w1_up, w2_up, learningRate)
# 
#        plt.subplot(121)
#        plt.imshow(x_train[i])
#        plt.subplot(122)
#        plt.imshow(dec[:-1].reshape(28,28)*255)
#        
#        plt.show(block = False)
#        #plt.pause(0.1)
#        plt.close()
#    loss = loss/N
#
#        
#    avg_losses[ep] = loss
#    print(loss)



#k=0
#slika = x_train[k]
#slika_flat = np.append(x_train[k].flatten(), 1).reshape(-1,1)/255
#decoded = forwardPassToDecoded(forwardPassToEncoded(slika_flat, w1), w2)
#plt.imshow(decoded[:-1].reshape(28,28)*255)
#plt.show() #block = False)


#print training history:
#plt.plot(avg_losses, 'g^')
#plt.show()






    