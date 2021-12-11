import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn

def transform_function():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def prediction(model,img):
    model = model.to('cpu')
    model.eval()
    image = Image.open(img, mode='r')
    image = image.convert('RGB')
    transform = transform_function()
    image = transform(image)
    sigmoid = nn.Sigmoid()
    outputs = model(image.unsqueeze(0))
    sigmoid_outputs = sigmoid(outputs).detach().squeeze().numpy()
    res = {'dog':sigmoid_outputs[0],'cat':sigmoid_outputs[1]}
    print(res)
    return res

def plot_prediction(prediction):
    fig = plt.figure()
    bars = plt.bar(range(len(prediction)), list(prediction.values()), align='center')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x(), yval + .005, yval)
    plt.xticks(range(len(prediction)), list(prediction.keys()))

    plt.ylabel("prediction")
    plt.yscale('log')
    plt.show()

def plot_loss(train_loss, val_loss, val_accuracy,name = "VGG"):
    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.plot(val_loss, label='validation loss')
    plt.plot(val_accuracy, label='validation accuracy')

    plt.title(name)
    plt.xlabel('epoch')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.legend()
    plt.savefig('./plot/'+name+'_loss.png')
    plt.show()


def plot_all_loss(vgg,vgg_BN,alex,alex_BN, loss = 'Training'):
    plt.figure()
    plt.plot(vgg, label='Vanilla VGG')
    plt.plot(vgg_BN, label='VGG with BN')
    plt.plot(alex, label='Vanilla AlexNet')
    plt.plot(alex_BN, label='AlexNet with BN')

    plt.title(loss + " Comparison")
    plt.xlabel('epoch')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(2))

    plt.legend()
    plt.savefig('./plot/'+loss.replace(" ","_") + '_compare.png')
    plt.show()

