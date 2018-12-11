from Core import *
import numpy
import scipy.special
import matplotlib.pyplot

class NumberDemo:
    def run(self):
        network = NeuralNetwork(784,100,10, 0.3)

        print("Read training dataset")
        file = open("mnist_train_100.csv")

        for line in  file.readlines():
    
            trainingValues = line.split(",")
            #image_bytes = numpy.asfarray(trainingValues[1:]).reshape((28,28))
            #matplotlib.pyplot.imshow(image_bytes, cmap= 'Greys', interpolation=
            #'None')
            #matplotlib.pyplot.show()
            scale = (numpy.asfarray(trainingValues[1:]) / 255.0 * 0.99) + 0.01
            requiredNumber = 10
            target = numpy.zeros(requiredNumber) + 0.01
            target[int(trainingValues[0])] = 0.99

            network.train(scale, target)
        pass
        file.close()

        print("Read real dataset")
        datasets = open("mnist_test_10.csv").readlines()[0].split(",")

        print("Expect" + datasets[0])
        result = network.query((numpy.asfarray(datasets[1:]) / 255.0 * 0.99) + 0.01)

        print(result)
    pass