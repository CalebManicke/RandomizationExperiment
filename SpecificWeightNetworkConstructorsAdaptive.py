#Network constructors for the adaptive black-box attack 
import torch.nn
import torch.nn.functional as F

class CustomWeights:
    def All_Zeros(layer): torch.nn.init.uniform_(layer.weight, 0.0, 0.0) 

    def All_Ones(layer): torch.nn.init.uniform_(layer.weight, 1.0, 1.0) 

    def Normal_Initialization(layer): torch.nn.init.normal_(layer.weight, mean=0, std=0.5) 

    def Uniform_Initialization(layer): torch.nn.init.uniform_(layer.weight)

    def Sparce_Initialization(layer): torch.nn.init.sparse_(layer.weight, sparsity=0.1)

    def Xavier_Initialization(layer): torch.nn.init.xavier_normal_(layer.weight)

    def Kaiming_Initialization(layer): torch.nn.init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu") 

class CarliniNetwork(torch.nn.Module):
    def __init__(self, inputImageSize, numClasses=10):
        super(CarliniNetwork, self).__init__()
        #Parameters for the network 
        params=[64, 64, 128, 128, 256, 256]
        #Create the layers 
        #model.add(Conv2D(params[0], (3, 3), input_shape=inputShape))
        #model.add(Activation('relu'))
        self.conv0 = torch.nn.Conv2d(in_channels=3, out_channels=params[0], kernel_size = (3,3), stride = 1)
        #model.add(Conv2D(params[1], (3, 3)))
        #model.add(Activation('relu'))
        self.conv1 = torch.nn.Conv2d(in_channels=params[0], out_channels=params[1], kernel_size = (3,3), stride = 1)
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        self.mp0 = torch.nn.MaxPool2d(kernel_size=(2,2))
        #model.add(Conv2D(params[2], (3, 3)))
        #model.add(Activation('relu'))
        self.conv2 = torch.nn.Conv2d(in_channels=params[1], out_channels=params[2], kernel_size = (3,3), stride = 1)
        #model.add(Conv2D(params[3], (3, 3)))
        #model.add(Activation('relu'))
        self.conv3 = torch.nn.Conv2d(in_channels=params[2], out_channels=params[3], kernel_size = (3,3), stride = 1)
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2,2))
        #model.add(Flatten())
        #model.add(Dense(params[4]))
        #model.add(Activation('relu'))
        #Next is flatten but we don't know the dimension size yet so must compute
        testInput = torch.zeros((1, 3, inputImageSize, inputImageSize))
        outputShape = self.figureOutFlattenShape(testInput)
        self.forward0 = torch.nn.Linear(in_features=outputShape[1], out_features=params[4]) #fix later 
        #model.add(Dropout(0.5))
        self.drop0 = torch.nn.Dropout(0.5)
        #model.add(Dense(params[5]))
        #model.add(Activation('relu'))
        self.forward1 = torch.nn.Linear(in_features=params[4], out_features=params[5])
        #model.add(Dense(numClasses, name="dense_2"))
        #model.add(Activation('softmax'))
        self.forward2 = torch.nn.Linear(in_features=params[5], out_features=numClasses)

    def initializeWeights(self, weightInitializationMethod):
        weightInitizationMethods = {'All_Zeros': (lambda x: CustomWeights.All_Zeros(x)), 
                                    'All_Ones': (lambda x: CustomWeights.All_Ones(x)), 
                                    'Uniform_Initialization': (lambda x: CustomWeights.Uniform_Initialization(x)), 
                                    'Normal_Initialization': (lambda x: CustomWeights.Normal_Initialization(x)), 
                                    'Sparse_Initialization': (lambda x: CustomWeights.Sparce_Initialization(x)),
                                    'Xavier_Initialization': (lambda x: CustomWeights.Xavier_Initialization(x)),
                                    'Kaiming_Initialization': (lambda x: CustomWeights.Kaiming_Initialization(x))}
        weightInitizationMethods[weightInitializationMethod](self.conv0)
        weightInitizationMethods[weightInitializationMethod](self.conv1)
        weightInitizationMethods[weightInitializationMethod](self.conv2)
        weightInitizationMethods[weightInitializationMethod](self.conv3)
        weightInitizationMethods[weightInitializationMethod](self.forward0)
        weightInitizationMethods[weightInitializationMethod](self.forward1)
        weightInitizationMethods[weightInitializationMethod](self.forward2)
        print(weightInitializationMethod + " Completed!")

    def forward(self, x):
        out = F.relu(self.conv0(x)) #model.add(Conv2D(params[0], (3, 3), input_shape=inputShape)) #model.add(Activation('relu'))
        out = F.relu(self.conv1(out)) #model.add(Conv2D(params[1], (3, 3))) #model.add(Activation('relu'))
        out = self.mp0(out)  #model.add(MaxPooling2D(pool_size=(2, 2)))
        out = F.relu(self.conv2(out)) #model.add(Conv2D(params[2], (3, 3)))  #model.add(Activation('relu'))
        out = F.relu(self.conv3(out)) #model.add(Conv2D(params[3], (3, 3))) #model.add(Activation('relu'))
        out = self.mp1(out) #model.add(MaxPooling2D(pool_size=(2, 2)))
        #print(out.shape)
        out = out.view(out.size(0), -1) #model.add(Flatten()), out.size(0) 
        #print(out.shape)
        out =  F.relu(self.forward0(out)) #model.add(Dense(params[4])) #model.add(Activation('relu'))
        out = self.drop0(out) #model.add(Dropout(0.5))
        out = F.relu(self.forward1(out)) #model.add(Dense(params[5])) #model.add(Activation('relu'))
        # Need to specify which dimension softmax layer adds up
        #print(out.shape)
        #out = F.softmax(self.forward2(out)) #model.add(Dense(numClasses, name="dense_2")) #model.add(Activation('softmax'))
        out = self.forward2(out)
        #print(out.shape)
        return out

    #This method is used to figure out what the input to the feedfoward part of the network should be 
    #We have to do this because Pytorch decided not to give this built in functionality for some reason 
    def figureOutFlattenShape(self, x):
        out = F.relu(self.conv0(x)) #model.add(Conv2D(params[0], (3, 3), input_shape=inputShape)) #model.add(Activation('relu'))
        out = F.relu(self.conv1(out)) #model.add(Conv2D(params[1], (3, 3))) #model.add(Activation('relu'))
        out = self.mp0(out)  #model.add(MaxPooling2D(pool_size=(2, 2)))
        out = F.relu(self.conv2(out)) #model.add(Conv2D(params[2], (3, 3)))  #model.add(Activation('relu'))
        out = F.relu(self.conv3(out)) #model.add(Conv2D(params[3], (3, 3))) #model.add(Activation('relu'))
        out = self.mp1(out) #model.add(MaxPooling2D(pool_size=(2, 2)))
        out = out.view(out.size(0), -1) #model.add(Flatten())
        return out.shape
