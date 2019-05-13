from share import *

# get user input value
parser = argparse.ArgumentParser()
parser.add_argument('data_dir')
parser.add_argument('--save_dir')
parser.add_argument('--arch')
parser.add_argument('--learning_rate')
parser.add_argument('--hidden_units')
parser.add_argument('--epochs')
parser.add_argument('--gpu')
args = parser.parse_args()

data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
learning_rate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.gpu

# user did not provide value, set default
if (arch == "vgg13"):
    input_size = 25088
    output_size = 102
elif (arch == "densenet121"):
    input_size = 1024
    output_size = 102
else:
    print("Please select model architectures vgg13 or densenet121.")
    exit()

if save_dir is None:
    save_dir = "save_checkpoint.pth"
    
if learning_rate is None:
    learning_rate = 0.001
else:
    learning_rate = float(learning_rate)

if hidden_units is None:
    if (arch == "vgg13"):
        hidden_units = 4096
    elif (arch == "densenet121"):
        hidden_units = 500
else:
    hidden_units = int(hidden_units)
    
if epochs is None:
    epochs = 25
else:
    epochs = int(epochs)

if device is None:
    device = "cpu"
    
if(data_dir == None) or (save_dir == None) or (arch == None) or (learning_rate == None) or (hidden_units == None) or (epochs == None) or (device == None):
    print("data_dir, arch , learning_rate, hidden_units, and epochs cannot be none")
    exit()


# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'valid', 'test']} 

# TODO: Build and train your network
if (arch == 'vgg13'):
    model = models.vgg13(pretrained=True)
elif (arch == 'densenet121'):
    model = models.densenet121(pretrained=True)
model

# TODO: Do validation on the test set
# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

# Build a feed-forward network
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                        ('relu', nn.ReLU()),
                                        ('dropout1',nn.Dropout(0.2)),
                                        ('fc2', nn.Linear(hidden_units, output_size)),
                                        ('output', nn.LogSoftmax(dim=1))]))

# Put the classifier on the pretrained network
model.classifier = classifier

# Train a model with a pre-trained network
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
#model.to('cuda')
model.to(device)
print("Start training model")
for e in range(epochs):

    for dataset in ['train', 'valid']:
        if dataset == 'train':
            model.train()  
        else:
            model.eval()   
        
        running_loss = 0.0
        running_accuracy = 0
        
        for inputs, labels in dataloaders[dataset]:
            #inputs, labels = inputs.to('cuda'), labels.to('cuda')
            inputs, labels = inputs.to(device), labels.to(device)
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
        
# Do validation on the test set
def check_accuracy_on_test(test_loader):    
    correct = 0
    total = 0
    #model.to('cuda:0')
    model.to(device)
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            #images, labels = images.to('cuda'), labels.to('cuda')
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
check_accuracy_on_test(dataloaders['train'])

# TODO: Save the checkpoint 
model.class_to_idx = image_datasets['train'].class_to_idx
model.cpu()
torch.save({'model': arch,
            'state_dict': model.state_dict(), 
            'class_to_idx': model.class_to_idx}, 
            save_dir)
print("Save model to:" + save_dir)