from yolo3 import yolo_body
from darknet53 import darknet_body
from resnet50 import ResNet50
from vgg16 import VGG16
from keras.layers import Input
from keras import models
from keras.utils import plot_model
from keras.layers import Lambda





##################################################恒等映射函数###################################################
# def my_identiy(inputs):
#     x=inputs
#     return x
#------------------------------------添加层的方式---------------------#
#my_identiy_layer1 = Lambda(my_identiy, name='my_identiy_layer1')(x)
#------------------------------------添加层的方式---------------------#
##################################################恒等映射函数###################################################


'''
#例1：
Inputs=Input([416,416,3])
net=ResNet50(input_tensor=Inputs)
print(net.output)
plot_model(net,show_shapes=True)
net.summary()
#``````````````````````````````````````````#'''




# '''
#例2：
Inputs=Input([416,416,3])
model=yolo_body(Inputs,3,7)
print('model.layers长度：',len(model.layers))
for i, layer in enumerate(model.layers):
    # print('层数：', i, '     层名称：', layer.name,'     该层输入形状：',layer.input.shape,'    该层输出形状：',layer.output.shape)
    print('层数：', i, '     层名称：', layer.name)
# plot_model(model,show_shapes=True)
model.summary()
# model.load_weights('../model_data/resnet50_imagenet.h5',by_name=True)
#```````````````````````````````````````````````#
# '''






'''
#例3：
Inputs=Input([416,416,3])
darknet53=models.Model(Inputs,darknet_body(Inputs))
print(darknet53.output)
plot_model(darknet53,show_shapes=True)
darknet53.summary()
'''




#VGG
'''Inputs=Input([416,416,3])
VGG16=models.Model(Inputs,VGG16(Inputs))
print(VGG16.output)
plot_model(VGG16,show_shapes=True)
print('model.layers长度：',len(VGG16.layers))
for i, layer in enumerate(VGG16.layers):
    # print('层数：', i, '     层名称：', layer.name,'     该层输入形状：',layer.input.shape,'    该层输出形状：',layer.output.shape)
    print('层数：', i, '     层名称：', layer.name)
VGG16.summary()'''



