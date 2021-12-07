import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from datasets import *
from model import *
import time
import copy
from utility import *
from PIL import Image


def train(model,name = "VGG"):
    train_loss_list = []
    val_loss_list = []
    val_accuracy_list = []

    since = time.time()
    num_epochs = 25
    criterion = nn.CrossEntropyLoss()

    # learning_rate = 0.001
    # weight_decay = 1e-5
    # optimizer = optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=weight_decay)

    optimizer = optim.SGD(list(model.parameters()), lr=1e-2, momentum=0.9, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_train_loss = 0.0
    best_val_loss = 0.0

    for epoch in range(num_epochs):
        print(name+' Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase

        # train
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward
            # track history if only in train
            torch.set_grad_enabled(True)
            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            # running_corrects += torch.sum(preds == labels.data)
        epoch_train_loss = running_loss / len(train_set)
        print('Epoch {:2d} Training Loss: {:.4f}'.format(epoch, epoch_train_loss))
        train_loss_list.append(epoch_train_loss)

        # validation process
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels in valloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # statistics
                val_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels.data).sum().item()
        epoch_val_loss = val_loss / len(val_set)
        val_loss_list.append(epoch_val_loss)

        epoch_acc = running_corrects / len(val_set)
        val_accuracy_list.append(epoch_acc)
        scheduler.step(epoch_val_loss)

        print('Epoch {:2d} Validation Loss: {:.4f} Acc: {:.4f}'.format(epoch,
            epoch_val_loss, epoch_acc))

        # deep copy the model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_train_loss = epoch_train_loss
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    torch.save(best_model_wts, name+'_best_model.pth')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print("best Accuracy: ", best_acc)
    print("best train loss: ", best_train_loss)
    print("best Validation loss: ", best_val_loss)
    print('Finished Training')

    # load best model weights
    model.load_state_dict(best_model_wts)

    plot_loss(train_loss_list, val_loss_list, val_accuracy_list,name)

    return train_loss_list,val_loss_list,val_accuracy_list

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transform_function()

    dataset = CatDogDataset(transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [20000, 5000])

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

    # train
    model_vgg = VGGnetwork()
    model_vgg_BN = VGGnetwork_BN()
    model_alex = AlexNet()
    model_alex_BN = AlexNet_BN()

    model_vgg = model_vgg.to(device)
    vgg_train_loss, vgg_val_loss, vgg_val_accuracy = train(model_vgg,"VGG")
    model_vgg = model_vgg.cpu()
    #
    # model_vgg_BN = model_vgg_BN.to(device)
    # vgg_BN_train_loss, vgg_BN_val_loss, vgg_BN_val_accuracy = train(model_vgg_BN,"VGG_BN")
    # model_vgg_BN = model_vgg_BN.cpu()

    # model_alex = model_alex.to(device)
    # alex_train_loss,alex_val_loss,alex_val_accuracy = train(model_alex,"AlexNet")
    # model_alex = model_alex.cpu()

    # model_alex_BN = model_alex_BN.to(device)
    # alex_BN_train_loss, alex_BN_val_loss, alex_BN_val_accuracy = train(model_alex_BN,"AlexNet_BN")
    # model_alex_BN = model_alex_BN.cpu()

    # plot_all_loss(vgg_train_loss,vgg_BN_train_loss,
    #                     alex_train_loss,alex_BN_train_loss,'Training loss')
    # plot_all_loss(vgg_val_loss, vgg_BN_val_loss,
    #                     alex_val_loss, alex_BN_val_loss, 'Validation loss')
    # plot_all_loss(vgg_val_accuracy, vgg_BN_val_accuracy,
    #                     alex_val_accuracy, alex_BN_val_accuracy, 'Validation Accuracy')

    model_vgg.load_state_dict(torch.load('./saved_model/VGG_best_model.pth'))
    model_alex.load_state_dict(torch.load('./saved_model/AlexNet_best_model.pth'))
