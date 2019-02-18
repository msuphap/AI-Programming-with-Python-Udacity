""" Developing an AI application
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset of 102 flower categories, you can see a few examples below.
The project is broken down into multiple steps:

Load and preprocess the image dataset
Train the image classifier on your dataset
Use the trained classifier to predict image content
We'll lead you through each part which you'll implement in Python.

When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.

First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

Please make sure if you are running this notebook in the workspace that you have chosen GPU rather than CPU mode.
"""
# Imports here
import time
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torch import optim
from collections import OrderedDict
from torchvision import datasets, transforms, models

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

"""
Load the data
Here you'll use torchvision to load the data (documentation). The data should be included alongside this notebook, otherwise you can download it here. The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.

The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.

The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225], calculated from the ImageNet images. These values will shift each color channel to be centered at 0 and range from -1 to 1.
"""
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {'train': transforms.Compose([transforms.RandomRotation(30),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'valid': transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                   'test': transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                   }

# TODO: Load the datasets with ImageFolder
directories = {'train': train_dir, 
               'valid': valid_dir, 
               'test' : test_dir}

image_datasets = {x: datasets.ImageFolder(directories[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']} 

"""
Label mapping
You'll also need to load in a mapping from category label to category name. You can find this in the file cat_to_name.json. It's a JSON object which you can read in with the json module. This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.
"""

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

"""
Building and training the classifier
Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from torchvision.models to get the image features. Build and train a new feed-forward classifier using those features.

We're going to leave this part up to you. Refer to the rubric for guidance on successfully completing this section. Things you'll need to do:

Load a pre-trained network (If you need a starting point, the VGG networks work great and are straightforward to use)
Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
Train the classifier layers using backpropagation using the pre-trained network to get the features
Track the loss and accuracy on the validation set to determine the best hyperparameters
We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!

When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

One last important tip if you're using the workspace to run your code: To avoid having your workspace disconnect during the long-running tasks in this notebook, please read in the earlier page in this lesson called Intro to GPU Workspaces about Keeping Your Session Active. You'll want to include code from the workspace_utils.py module.
"""
# TODO: Build and train your network
model = models.vgg13(pretrained=True)
model

"""
Downloading: "https://download.pytorch.org/models/vgg13-c768596a.pth" to /root/.torch/models/vgg13-c768596a.pth
100%|██████████| 532194478/532194478 [00:26<00:00, 20341286.29it/s]

VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (16): ReLU(inplace)
    (17): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace)
    (19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (20): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): ReLU(inplace)
    (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (23): ReLU(inplace)
    (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    (1): ReLU(inplace)
    (2): Dropout(p=0.5)
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    (4): ReLU(inplace)
    (5): Dropout(p=0.5)
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)
"""

"""
Testing your network
It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.
"""

# TODO: Do validation on the test set
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Build a feed-forward network
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                        ('relu', nn.ReLU()),
                                        ('dropout1',nn.Dropout(0.2)),
                                        ('fc2', nn.Linear(4096, 102)),
                                        ('output', nn.LogSoftmax(dim=1))]))

# Put the classifier on the pretrained network
model.classifier = classifier

# Train a model with a pre-trained network
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

epochs = 25
model.to('cuda')

for e in range(epochs):

    for dataset in ['train', 'valid']:
        if dataset == 'train':
            model.train()  
        else:
            model.eval()   
        
        running_loss = 0.0
        running_accuracy = 0
        
        for inputs, labels in dataloaders[dataset]:
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(dataset == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward 
                if dataset == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_accuracy += torch.sum(preds == labels.data)
        
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        epoch_loss = running_loss / dataset_sizes[dataset]
        epoch_accuracy = running_accuracy.double() / dataset_sizes[dataset]
        
        print("Epoch: {}/{}... ".format(e+1, epochs),
              "{} Loss: {:.4f}    Accurancy: {:.4f}".format(dataset, epoch_loss, epoch_accuracy))
        
"""
Epoch: 1/25...  train Loss: 2.4321    Accurancy: 0.4800
Epoch: 1/25...  valid Loss: 0.7083    Accurancy: 0.8068
Epoch: 2/25...  train Loss: 1.1893    Accurancy: 0.6836
Epoch: 2/25...  valid Loss: 0.6413    Accurancy: 0.8289
Epoch: 3/25...  train Loss: 1.0101    Accurancy: 0.7334
Epoch: 3/25...  valid Loss: 0.5341    Accurancy: 0.8496
Epoch: 4/25...  train Loss: 0.9374    Accurancy: 0.7500
Epoch: 4/25...  valid Loss: 0.5510    Accurancy: 0.8667
Epoch: 5/25...  train Loss: 0.9079    Accurancy: 0.7636
Epoch: 5/25...  valid Loss: 0.5527    Accurancy: 0.8741
Epoch: 6/25...  train Loss: 0.8795    Accurancy: 0.7766
Epoch: 6/25...  valid Loss: 0.5254    Accurancy: 0.8949
Epoch: 7/25...  train Loss: 0.8459    Accurancy: 0.7862
Epoch: 7/25...  valid Loss: 0.6391    Accurancy: 0.8606
Epoch: 8/25...  train Loss: 0.8698    Accurancy: 0.7767
Epoch: 8/25...  valid Loss: 0.5165    Accurancy: 0.8826
Epoch: 9/25...  train Loss: 0.8382    Accurancy: 0.7842
Epoch: 9/25...  valid Loss: 0.6678    Accurancy: 0.8765
Epoch: 10/25...  train Loss: 0.8145    Accurancy: 0.7961
Epoch: 10/25...  valid Loss: 0.6172    Accurancy: 0.8826
Epoch: 11/25...  train Loss: 0.7770    Accurancy: 0.8092
Epoch: 11/25...  valid Loss: 0.5726    Accurancy: 0.8900
Epoch: 12/25...  train Loss: 0.8356    Accurancy: 0.7985
Epoch: 12/25...  valid Loss: 0.6131    Accurancy: 0.8839
Epoch: 13/25...  train Loss: 0.7682    Accurancy: 0.8135
Epoch: 13/25...  valid Loss: 0.6471    Accurancy: 0.8936
Epoch: 14/25...  train Loss: 0.7590    Accurancy: 0.8196
Epoch: 14/25...  valid Loss: 0.6547    Accurancy: 0.9034
Epoch: 15/25...  train Loss: 0.7597    Accurancy: 0.8161
Epoch: 15/25...  valid Loss: 0.6457    Accurancy: 0.8826
Epoch: 16/25...  train Loss: 0.7927    Accurancy: 0.8117
Epoch: 16/25...  valid Loss: 0.7846    Accurancy: 0.8851
Epoch: 17/25...  train Loss: 0.7908    Accurancy: 0.8190
Epoch: 17/25...  valid Loss: 0.7144    Accurancy: 0.8729
Epoch: 18/25...  train Loss: 0.7733    Accurancy: 0.8139
Epoch: 18/25...  valid Loss: 0.6588    Accurancy: 0.8998
Epoch: 19/25...  train Loss: 0.7667    Accurancy: 0.8190
Epoch: 19/25...  valid Loss: 0.7369    Accurancy: 0.9010
Epoch: 20/25...  train Loss: 0.8096    Accurancy: 0.8208
Epoch: 20/25...  valid Loss: 0.7336    Accurancy: 0.8839
Epoch: 21/25...  train Loss: 0.7314    Accurancy: 0.8312
Epoch: 21/25...  valid Loss: 0.6973    Accurancy: 0.8973
Epoch: 22/25...  train Loss: 0.7436    Accurancy: 0.8263
Epoch: 22/25...  valid Loss: 0.6775    Accurancy: 0.9120
Epoch: 23/25...  train Loss: 0.7469    Accurancy: 0.8271
Epoch: 23/25...  valid Loss: 0.6763    Accurancy: 0.9071
Epoch: 24/25...  train Loss: 0.6976    Accurancy: 0.8379
Epoch: 24/25...  valid Loss: 0.7526    Accurancy: 0.8998
Epoch: 25/25...  train Loss: 0.7437    Accurancy: 0.8312
Epoch: 25/25...  valid Loss: 0.6901    Accurancy: 0.9022
"""


# Do validation on the test set
def check_accuracy_on_test(test_loader):    
    correct = 0
    total = 0
    model.to('cuda')
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
check_accuracy_on_test(dataloaders['train'])
"""
Accuracy of the network on the test images: 87 %
"""

"""
Save the checkpoint
Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: image_datasets['train'].class_to_idx. You can attach this to the model as an attribute which makes inference easier later on.

model.class_to_idx = image_datasets['train'].class_to_idx

Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, optimizer.state_dict. You'll likely want to use this trained model in the next part of the project, so best to save it now.

"""

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'model': 'vgg13',
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            'save_checkpoint.pth')

"""
Loading the checkpoint
At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.
"""

# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_model(checkpoint_path):
    
    check_path = torch.load(checkpoint_path)
    
    model = models.vgg13(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = check_path['class_to_idx']
    
    # Build a feed-forward network
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(25088, 4096)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(4096, 102)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(check_path['state_dict'])
    
    return model

model = loading_model('save_checkpoint.pth')
model

"""
VGG(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU(inplace)
    (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (3): ReLU(inplace)
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (6): ReLU(inplace)
    (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (8): ReLU(inplace)
    (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): ReLU(inplace)
    (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (13): ReLU(inplace)
    (14): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (15): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (16): ReLU(inplace)
    (17): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): ReLU(inplace)
    (19): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (20): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): ReLU(inplace)
    (22): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (23): ReLU(inplace)
    (24): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (classifier): Sequential(
    (fc1): Linear(in_features=25088, out_features=4096, bias=True)
    (relu): ReLU()
    (dropout1): Dropout(p=0.2)
    (fc2): Linear(in_features=4096, out_features=102, bias=True)
    (output): LogSoftmax()
  )
)
"""
"""
Inference for classification
Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called predict that takes an image and a model, then returns the top  𝐾  most likely classes along with the probabilities. It should look like

probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
First you'll need to handle processing the input image such that it can be used in your network.

Image Preprocessing
You'll want to use PIL to load the image (documentation). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training.

First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the thumbnail or resize methods. Then you'll need to crop out the center 224x224 portion of the image.

Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so np_image = np.array(pil_image).

As before, the network expects the images to be normalized in a specific way. For the means, it's [0.485, 0.456, 0.406] and for the standard deviations [0.229, 0.224, 0.225]. You'll want to subtract the means from each color channel, then divide by the standard deviation.

And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using ndarray.transpose. The color channel needs to be first and retain the order of the other two dimensions.
"""

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    
    # Edit
    edit_image = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # Dimension
    img_tensor = edit_image(pil_image)
    processed_image = np.array(img_tensor)
    processed_image = processed_image.transpose((0, 2, 1))
    
    return processed_image

# Test image after process
image_path = 'flowers/test/1/image_06743.jpg'
img = process_image(image_path)
print(img.shape) 
"""
(3, 224, 224)
"""

"""
To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your process_image function works, running the output through this function should return the original image (except for the cropped out portions).
"""

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
  
  imshow(process_image("flowers/test/1/image_06752.jpg"))
  
  """
  Class Prediction
Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top- 𝐾 ) most probable classes. You'll want to calculate the class probabilities then find the  𝐾  largest values.

To get the top  𝐾  largest values in a tensor use x.topk(k). This method returns both the highest k probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using class_to_idx which hopefully you added to the model or from an ImageFolder you used to load the data (see here). Make sure to invert the dictionary so you get a mapping from index to class as well.

Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.

probs, classes = predict(image_path, model)
print(probs)
print(classes)
> [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
> ['70', '3', '45', '62', '55']
"""
  
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.class_to_idx = image_datasets['train'].class_to_idx
    model.to('cuda')
    img_torch = process_image(image_path)
    img_torch = torch.from_numpy(img_torch).type(torch.FloatTensor)
    img_torch = img_torch.unsqueeze(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())

    probability = F.softmax(output.data,dim=1)
    probabilies = probability.topk(topk)
    score = np.array(probabilies[0][0])
    index = 1
    flowers_list = [cat_to_name[str(index + 1)] for index in np.array(probabilies[1][0])]
   
    return score, flowers_list
   
"""
Sanity Checking
Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use matplotlib to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:

You can convert from the class integer encoding to actual flower names with the cat_to_name.json file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the imshow function defined above.
"""

# TODO: Display an image along with the top 5 classes
def display_top5(image_path, model):
    
    # Setting plot area
    plt.figure(figsize = (3,6))
    ax = plt.subplot(2,1,1)
    
    # Display test flower
    img = process_image(image_path)
    get_title  = image_path.split('/')
    print(cat_to_name[get_title[2]])
    imshow(img, ax, title = cat_to_name[get_title[2]]);
    
    # Making prediction
    score, flowers_list = predict(image_path, model) 
    fig,ax = plt.subplots(figsize=(4,3))
    sticks = np.arange(len(flowers_list))
    ax.barh(sticks, score, height=0.3, linewidth=2.0, align = 'center')
    ax.set_yticks(ticks = sticks)
    ax.set_yticklabels(flowers_list)

  
image_path = 'flowers/test/28/image_05277.jpg'
display_top5(image_path, model)


image_path = 'flowers/test/1/image_06752.jpg'
display_top5(image_path, model)


