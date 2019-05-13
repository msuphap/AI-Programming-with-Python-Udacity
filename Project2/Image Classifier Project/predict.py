from share import *

parser = argparse.ArgumentParser()
parser.add_argument('image_path')
parser.add_argument('checkpoint')
parser.add_argument('--top_k')
parser.add_argument('--category_names')
parser.add_argument('--gpu')
args = parser.parse_args()

image_path = args.image_path
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names
device = args.gpu

# user did not provide value, set default
if top_k is None:
    top_k = 5
    
if device is None:
    device = "cpu"

if category_names is None:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
else:
    filename, file_extension = os.path.splitext(category_names)
    if file_extension != '.json':
        print("Please use file extension .json instead of " + category_names + ".")
        exit()
    else:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def loading_model(checkpoint_path):
    
    check_path = torch.load(checkpoint_path)
    # add for test
    arch = "vgg13"
    
    if (arch == 'vgg13'):
        model = models.vgg13(pretrained=True)
        input_size = 25088
        hidden_units = 4096
        output_size = 102
    elif (arch == 'densenet121'):
        model = models.densenet121(pretrained=True)
        input_size = 1024
        hidden_units = 500
        output_size = 102
            
    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = check_path['class_to_idx']
                    
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(input_size, hidden_units)),
                                            ('relu', nn.ReLU()),
                                            ('dropout1',nn.Dropout(0.2)),
                                            ('fc2', nn.Linear(hidden_units, output_size)),
                                            ('output', nn.LogSoftmax(dim=1))]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(check_path['state_dict'])
    ####print("The model is loaded to" + save_dir)
    return model

model = loading_model('save_checkpoint.pth')

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

#image_path = 'flowers/test/1/image_06752.jpg'
#img = process_image(image_path)
#print(img.shape)

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

#imshow(process_image("flowers/test/1/image_06752.jpg"))

def predict(image_path, model, topk = top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.class_to_idx = image_datasets['train'].class_to_idx
    #####model.to('cuda:0')
    model.to(device)
    img_torch = process_image(image_path)
    img_torch = torch.from_numpy(img_torch).type(torch.FloatTensor)
    img_torch = img_torch.unsqueeze(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        if device == "cpu":
            output = model.forward(img_torch.cpu())
        elif device == "cuda":
            output = model.forward(img_torch.cuda())
   
    probability = F.softmax(output.data,dim=1)
    probabilies = probability.topk(topk)
    score = np.array(probabilies[0][0])
    index = 1
    flowers_list = [cat_to_name[str(index + 1)] for index in np.array(probabilies[1][0])]
   
    return score, flowers_list

# TODO: Display an image along with the top 5 classes
def display_top(image_path, model, save_result_dir):
   
    # Setting plot area
    plt.figure(figsize = (3,6))
    ax = plt.subplot(1,1,1)
    # Display test flower
    img = process_image(image_path)
    get_title  = image_path.split('/')
    print(cat_to_name[get_title[2]])
    
    imshow(img, ax, title = cat_to_name[get_title[2]]);
    
    # Making prediction
    score, flowers_list = predict(image_path, model) 
    fig,ax = plt.subplots(figsize=(10,10))
    sticks = np.arange(len(flowers_list))
    ax.barh(sticks, score, height=0.3, linewidth=2.0, align = 'center')
    ax.set_yticks(ticks = sticks)
    ax.set_yticklabels(flowers_list)
    #plt.show()
    plt.savefig(save_result_dir)

image_path = 'flowers/test/28/image_05277.jpg'
get_title  = image_path.split('/')
print("Test image:" + cat_to_name[get_title[2]])
save_result_dir = 'save_prediction_result_1'
display_top(image_path, model, save_result_dir)
print("Save prediction result to:" + save_result_dir)
print("Prediction result:")
score, flower_list = predict(image_path, model)
print(flower_list)
print(np.exp(score))
print("-------------------------------------------")
image_path = 'flowers/test/1/image_06752.jpg'
get_title  = image_path.split('/')
print("Test image:" + cat_to_name[get_title[2]])
save_result_dir = 'save_prediction_result_2'
display_top(image_path, model, save_result_dir)
print("Save prediction result to:" + save_result_dir)
print("Prediction result:")
score, flower_list = predict(image_path, model)
print(flower_list)
print(np.exp(score))
print("-------------------------------------------")
