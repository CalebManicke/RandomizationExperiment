import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from random import shuffle
import DataManagerPytorch as datamanager
import SpecificWeightNetworkConstructorsAdaptive as networks
import matplotlib.pyplot as plt
import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main():
    # Batch size, this is variable for SGD experiment!!!
    # Else set to 128!!!
    batchSize = 128

    # Fixed hyperparameters
    imgSize = 32
    num_classes = 10
    learning_rate = 0.0001
    num_epochs = 50

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    if (torch.cuda.is_available()):
        print('Number of CUDA Devices:', torch.cuda.device_count())
        print('CUDA Device Name:',torch.cuda.get_device_name(0))
        print('CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

    # Initialize validation & training data loaders
    transformTrain = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    transformTest = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    cifar10_train_loader = datamanager.GetCIFAR10Training(imgSize, batchSize)
    cifar10_val_loader = datamanager.GetCIFAR10Validation(imgSize, batchSize)

    # Initialize carlini network 
    cifar_10_network = networks.CarliniNetwork(imgSize, num_classes).to(device)

    # Specify weight initialization method to 'All_Zeros', 'All_Ones', 'Uniform_Initialization', 'Normal_Initialization', 'Xavier_Initialization', 'Kaiming_Initialization'
    # If performing batch size variable experiment, we can just comment this out
    weightInitializationMethod = "Kaiming_Initialization"
    cifar_10_network.initializeWeights(weightInitializationMethod)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cifar_10_network.parameters(), lr = learning_rate)

    # Train and validate
    trainingLoss = train(num_epochs, num_classes, cifar_10_network, cifar10_train_loader, device, optimizer, criterion)
    print("------------------------------------")
    print("After Training: ")  
    print("------------------------------------")
    train_acc = get_train_acc(cifar10_train_loader, cifar_10_network, device)
    val_acc = get_val_acc(cifar10_val_loader, cifar_10_network, device)

    # Create plot of training loss vs epoch
    epochRange = [epoch for epoch in range(num_epochs)]
    plt.plot(epochRange, trainingLoss, 'r--')
    plt.xlabel('Training Epochs')
    plt.ylabel('Training Loss')
    plt.show()
    saveDir = os.getcwd() + "//Randomized_Carlini_Weights_Experiment//"
    if not os.path.exists(saveDir): os.makedirs(saveDir)
    plt.savefig(saveDir + weightInitializationMethod + ".png")
    plt.close()

    # Save trained model
    saveTag = "Randomized_Carlini_Weights_Experiment//" + weightInitializationMethod + ".th"
    torch.save({'epoch': num_epochs, 'state_dict': cifar_10_network.state_dict(), 'bestAcc': val_acc}, os.path.join(os.getcwd(), saveTag)) 


# Train network
def train(num_epochs, num_classes, cifar_10_network, cifar10_train_loader, device, optimizer, criterion):
    cifar_10_network.train()
    trainingLoss = []

    for epoch in range(num_epochs): 
        print("------------------------------------")
        print("Epoch: ", epoch+1)  
        print("------------------------------------")
        totalLoss = 0
        for i, (data, targets) in enumerate(cifar10_train_loader):      # for i, (data, targets) in enumerate(cifar10_train_loader):
            data = Variable(data.to(device = device), requires_grad = True)
            targets = targets.to(device = device)
            target_vars = targets
        
            # Forward pass
            scores = cifar_10_network(data)
            loss = criterion(scores, target_vars).float()   # Loss function
            # Record total training loss over epoch
            totalLoss += loss.item()

            # Backward
            optimizer.zero_grad()       # We want to set all gradients to zero for each batch so it doesn't store backprop calculations from previous forwardprops
            loss.backward()
            optimizer.step()

            print("Step: ", i+1, "Loss: ", loss.item())
        # Save total training loss over epoch
        trainingLoss.append(totalLoss)
    return trainingLoss
               

def get_train_acc(cifar10_train_loader, cifar_10_network, device):  
    acc = datamanager.validateD(cifar10_train_loader, cifar_10_network, device)
    print("Training Set Accuracy: ", acc)

def get_val_acc(cifar10_val_loader, cifar_10_network, device):  
    acc = datamanager.validateD(cifar10_val_loader, cifar_10_network, device)
    print("Validation Set Accuracy: ", acc)
    
def saveCheckpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

# We set num_workers = 1 in DataManagersPyTorch, so to avoid an error b/c of multithreading in Windows, we use if-clause protection
if __name__ == '__main__':
    main()