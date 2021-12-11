from model import *
from utility import *

model_vgg = VGGnetwork()
model_alex = AlexNet()
model_vgg.load_state_dict(torch.load('./saved_model/VGG_best_model.pth'))
model_alex.load_state_dict(torch.load('./saved_model/AlexNet_best_model.pth'))

image_num = 581
img_path = './data/test/' + str(image_num) + '.jpg'
a= prediction(model_vgg,img_path)
b=prediction(model_alex,img_path)

plot_prediction(a)
plot_prediction(b)