import numpy
# scipy.special for the sigmoid function expit(), and its inverse logit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are insi

class NeuralNetwork:
    def __init__(self,inputnodes, hiddennodes, outputnodes, learningrate):                
        self.InputNodes = inputnodes
        self.HiddenNodes = hiddennodes
        self.OutputNodes = outputnodes
        self.Learningrate = learningrate
        
        #Gewichte Eingang und Ausgang
        self.WightInputToHidden = numpy.random.normal(0.0, pow(self.HiddenNodes, -0.5), (self.HiddenNodes, self.InputNodes))
        self.WightHiddenToOutput = numpy.random.normal(0.0, pow(self.OutputNodes, -0.5), (self.OutputNodes, self.HiddenNodes))

        self.Activation_Function = lambda x: scipy.special.expit(x)

        pass
    def train(self, inputList, targetList):
        #redundant könnte man auch mit query aufrufen.  -> anmerkung an den
        #Author?
        input = numpy.array(inputList, ndmin=2).T

        hiddenInputs = numpy.dot(self.WightInputToHidden, input)
        hiddenOutputs = self.Activation_Function(hiddenInputs)
        finalInputs = numpy.dot(self.WightHiddenToOutput, hiddenOutputs)
        finalOutputs = self.Activation_Function(finalInputs)        

        targets = numpy.array(targetList, ndmin= 2).T
        outputDiff = targets - finalOutputs
        hiddenDiff = numpy.dot(self.WightHiddenToOutput.T, outputDiff)
        #update wights
        self.WightHiddenToOutput += self.Learningrate * numpy.dot((outputDiff * finalOutputs * (1.0 - finalOutputs)), numpy.transpose(hiddenOutputs))
        self.WightInputToHidden += self.Learningrate * numpy.dot((hiddenDiff * hiddenOutputs * (1.0 - hiddenOutputs)), numpy.transpose(input))

        pass
    def query(self, inputsList):

        input = numpy.array(inputsList, ndmin=2).T

        hiddenInputs = numpy.dot(self.WightInputToHidden, input)
        hiddenOutputs = self.Activation_Function(hiddenInputs)
        finalInputs = numpy.dot(self.WightHiddenToOutput, hiddenOutputs)
        finalOutputs = self.Activation_Function(finalInputs)        

        return finalOutputs
        pass