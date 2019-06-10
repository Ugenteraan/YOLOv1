#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import glob
import cv2
import xmltodict
import tensorflow as tf


# In[2]:


data_images_path     = 'VOCdevkit/VOC2012/JPEGImages'
data_annotation_path = 'VOCdevkit/VOC2012/Annotations'
image_height = 448
image_width  = 448
image_depth  = 3
learning_rate = 1e-4
#constants
lambda_coord = 5
lambda_noobj = 0.5
epsilon = 1e-9
batch_size = 50
epoch = 1000

model_save_path = os.getcwd() + '/model/model.ckpt'

# In[3]:


#since the naming of the image files and annotation files of VOC Pascal Dataset differs only in the extension,
#sorting the lists would enable us to select an image file and its corresponding annotation file at the same 
#index position from the lists.
list_images      = sorted([x for x in glob.glob(data_images_path + '/**')])     #length : 17125
list_annotations = sorted([x for x in glob.glob(data_annotation_path + '/**')]) #length : 17125




def get_total_classes(xml_files=list_annotations):
    '''Get all the classes in the dataset to construct one hot vector later.
       Parameter
       ---------
       xml_files : a list containing paths to every single xml files.
    '''
    
    classes = []
    
    for file in xml_files: #iterate through every xml files
      
        f = open(file)
        doc = xmltodict.parse(f.read()) #parse the xml file
        
        #Some xml files may only contain one object tag as there's only 1 object in the image.
        #For-looping over these tags throws a TypeError. Therefore, we use try-except to avoid this.
        try:
            for obj in doc['annotation']['object']: # try iterating through the objects in the xml file
                    classes.append(obj['name'])
        
        except TypeError as e:
            classes.append(doc['annotation']['object']['name'])
        
        f.close()
            
    classes = list(set(classes)) #set to remove duplicates.
    classes.sort() #sort the list in ascending order
    
    
    return classes


# In[6]:


classes = get_total_classes()
print(classes)


# In[7]:


C = len(classes) #20
S = 7 #cells
B = 2 #num of bounding boxes per cell


# In[8]:


def get_label(xml_file_path):
    '''Reads one file's annotation information and convert it to YOLO format.
       Returns a label list for one image.
       Parameter 
       ---------
       xml_file_path : path to a Pascal VOC format xml file   | string
    '''
    
    f   = open(xml_file_path)
    doc = xmltodict.parse(f.read()) #parse the xml file and convert it into python dict
    
    height = doc['annotation']['size']['height']
    width  = doc['annotation']['size']['width']
    
    #Each image must have labels for every cell. This means that in our case, S=7, C=20, we need to have
    #[x,y,w,h,confidence, Pr(C_0),Pr(C_1), ... ,Pr(C_19)]. The length of the list would be 25. The confidence is
    #zero when there is no object in the particular cell. Otherwise, the confidence is equal to the IoU between
    #the predicted bounding box and the ground truth. Hence to calculate the confidence, in the fifth index of 
    #the list, we mark the Pr(object). If there is an object in the cell, Pr(object) = 1. 0 otherwise. When 
    #the network predicts the Pr(object), the predicted Pr(object) and the ground truth of Pr(object) will be 
    #used to calculate to calculate the confidence. NOTE that the length of the prediction is 30 since there are
    #2 bounding box predictions. During training, only one of the box will be selected based on IoU. Hence,
    #the label's list length is 25
    label = [[0.0] * (5+C)] * S**2 #a 2-D list of zeros length 49 (S**2) where each element in the list is a list
    #of zeros of length 25 (5 + C).
    
    
    #Some xml files may only contain one object tag as there's only 1 object in the image.
    #For-looping over these tags throws a TypeError. Therefore, we use try-except to avoid this.
    try:
        for obj in doc['annotation']['object']:#we have to iterate here since an img may contain more than 1 obj
            
            #retrieve the information from the xmldict
            name  = obj['name']
            x_min = obj['bndbox']['xmin']
            x_max = obj['bndbox']['xmax']
            y_min = obj['bndbox']['ymin']
            y_max = obj['bndbox']['ymax']

            #center of the box.
            center_x = int(float(x_max)) - int(float(x_min)) 
            center_y = int(float(y_max)) - int(float(y_min)) 

            #the width and height of each cell when we divide the image into S x S cells.
            cell_size_x = int(width)/S 
            cell_size_y = int(height)/S

            '''
            Quote from paper 
            ----------------
            If the center of an object falls into a grid cell, that grid cell is responsible for detecting
            that object.

            '''
            #get the cell that is responsible for the object and the value of the coordinates relative to 
            #the responsible grid cell.
            x_coord_box, x_in_cell = divmod(center_x, cell_size_x)
            y_coord_box, y_in_cell = divmod(center_y, cell_size_y)

            #normalize the x and y coordinates in the cell.
            x = x_in_cell/cell_size_x
            y = y_in_cell/cell_size_y

            #normalize the width and height of the bounding box relative to the entire image's width and height.
            w = (int(float(x_max)) - int(float(x_min)))/int(float(width))
            h = (int(float(y_max)) - int(float(y_min)))/int(float(height))
            
            #one-hot *list* for the class
            one_hot_list = [0] * C #A list of zeros at length C
            index = classes.index(name) #get the index of the class from the list 'classes'
            one_hot_list[index] = 1.0 
            
            #list for each object. Round the floats to 2 decimal places
            obj_info = [round(x,2),round(y,2),round(w,2),round(h,2), 1.0 ] + one_hot_list
            
            #since here we have the position of the box as a coordinate, we can convert that coordinate to box
            #number with (x-coor + (y-coor x 7)). This is assuming the box numbering is from left to right
            #starting from 0.
            box_position = x_coord_box + (y_coord_box * 7)
            label[int(box_position)] = obj_info #replace the list of zeros

    #Some xml files may only contain one object tag as there's only 1 object in the image.
    #For-looping over these tags throws a TypeError. Therefore, we use try-except to avoid this.
    except TypeError as e:
        
        #Note that we use the doc dictionary, not obj dictionary
        name  = doc['annotation']['object']['name']
        x_min = doc['annotation']['object']['bndbox']['xmin']
        x_max = doc['annotation']['object']['bndbox']['xmax']
        y_min = doc['annotation']['object']['bndbox']['ymin']
        y_max = doc['annotation']['object']['bndbox']['ymax']

        #center of the box.
        center_x = int(float(x_max)) - int(float(x_min)) 
        center_y = int(float(y_max)) - int(float(y_min)) 

        #the width and height of each cell when we divide the image into S x S cells.
        cell_size_x = int(width)/S 
        cell_size_y = int(height)/S

        '''
        Quote from paper 
        ----------------
        If the center of an object falls into a grid cell, that grid cell is responsible for detecting
        that object.

        '''
        #get the cell that is responsible for the object and the value of the coordinates relative to 
        #the responsible grid cell.
        x_coord_box, x_in_cell = divmod(center_x, cell_size_x)
        y_coord_box, y_in_cell = divmod(center_y, cell_size_y)

        #normalize the x and y coordinates in the cell.
        x = x_in_cell/cell_size_x
        y = y_in_cell/cell_size_y

        #normalize the width and height of the bounding box relative to the entire image's width and height.
        w = (int(float(x_max)) - int(float(x_min)))/int(float(width))
        h = (int(float(y_max)) - int(float(y_min)))/int(float(height))
        
        #one-hot *list* for the class
        one_hot_list = [0] * C #A list of zeros at length C
        index = classes.index(name) #get the index of the class from the list 'classes'
        one_hot_list[index] = 1.0 

        #list for each object. Round the floats to 2 decimal places
        obj_info = [round(x,2),round(y,2),round(w,2),round(h,2), 1.0 ] + one_hot_list
        
        #since here we have the position of the box as a coordinate, we can convert that coordinate to box
        #number with (x-coor + (y-coor x 7)). This is assuming the box numbering is from left to right
        #starting from 0.
        box_position = x_coord_box + (y_coord_box * 7)
        label[int(box_position)] = obj_info #replace the list of zeros
    
    f.close()
        
    return label #returns the label of an image


# In[9]:


def load_dataset(first_index, last_index):
    '''Load images into numpy array in a specific size (last_index - first_index).
       Load annotations in YOLO format.
       Returns np images and label
       Parameter
       ---------
       first_index : integer
       last_index  : integer
    '''
    
    images = [] #initialize an empty list to append the images
    labels    = [] #initialize an empty list to append the labels
    

    for i in range(first_index,last_index): 
        
        im = cv2.imread(list_images[i])                 #read the images from the path
        im = cv2.resize(im, (image_height,image_width)) #resize the images to 448x448x3
        images.append(im)                               #append the image into the list
        
        label = get_label(list_annotations[i]) #get the list label for an image 
        labels.append(label) #append a single label into the list of labels
        
        
    labels    = np.asarray(labels)    #convert the label list into np array
    images    = np.asarray(images) #convert the images list into np array
    
    return (images, labels)


# In[10]:


def bbox_selector(box1, box2, truth):
    '''Returns 3 tensors where the first two tensors are of values of either 1 or 0 based on which 
       bounding box should be selected and the third one is the IoU. The decision of which bounding box should 
       be selected is based on which predicted bounding box has the highest IoU with the ground truth.
       Parameters
       ----------
       box1 : first predicted bounding box  | [batch_size, 49, 4]
       box2 : second predicted bounding box | [batch_size, 49, 4]
       truth: ground truth np array         | [batch_size, 49, 4]
    '''
    
    #Since the ground truth labels and the predictions are normalized to the image size and the cell size, we
    #will have to convert them back to coordinates in the image to calculate the IoU.
    #Since the image has been resized to 448 x 488, we will use this as the height and width.
    
    #the width and height of each cell
    cell_size_x = image_width/S
    cell_size_y = image_height/S
    
    IoU_list = []
    
    #iterate through the cells
    for i in range(S**2):
        
        '''
        Ground truth coordinates
        '''
        
        #y_offset tells us how many times we had to minus 7 from the current box position. It gives us
        #the position of the box from the top while x_offset gives is the current box position from the left.
        y_offset, x_offset = divmod(i,S)
        
        #iterates through each cell to get the tground ruth bounding box's normalized x,y,w and h values
        truth_x_norm, truth_y_norm = truth[:,i,0], truth[:,i,1] #shape : [batch_size]
        truth_w_norm, truth_h_norm = truth[:,i,2], truth[:,i,3] #shape : [batch_size]
        
        #to get the x coordinate in the image, we have to check on which cell the points fall into.
        #If the point falls in the 5th cell from the left (ignore the y axis), then the x coordinate
        #is equal to (4 x width of a cell) + (normalized_x_value x width of a cell). Hence the modulus to
        #check how many cells from the left. Since the index starts from 0 we don't have to minus 1. As for y
        #coordinate in the image, we have to check from the top box. Hence the y_offset.
        truth_x = x_offset*cell_size_x + truth_x_norm*cell_size_x #shape : [batch_size]
        truth_y = y_offset*cell_size_y + truth_y_norm*cell_size_y #shape : [batch_size]
        
        #since the width and height are normalized based on the entire image, to get the width and height
        #we do : normalized width x image width = width. normalized height x image height = height.
        truth_w = image_width * truth_w_norm  #shape : [batch_size]
        truth_h = image_height * truth_h_norm #shape : [batch_size]
        
        #to calculate the IoU, it is easier to have the coordinate of the top-left and bottom-right of a box.
        #ground-truth bounding box.
       
        #the coordinates are the center of the box. Therefore we use half of the width and height to find
        #the box corner's coordinates.
        truth_top_left_coor_x = truth_x - truth_w/2 #x coordinate of top left corner shape: [batch_size]
        truth_top_left_coor_y = truth_y - truth_h/2 #y coordinate of top left corner shape: [batch_size]
        truth_btm_rght_coor_x = truth_x + truth_w/2 #x coordinate of btm right corner shape: [batch_size]
        truth_btm_rght_coor_y = truth_y + truth_h/2 #y coordinate of btm right corner shape: [batch_size]
        
        '''
        bounding box 1 coordinate
        '''
        
        box1_x_norm,box1_y_norm = box1[:,i,0], box1[:,i,1] 
        box1_w_norm,box1_h_norm = box1[:,i,2], box1[:,i,3]
        
        box1_x = x_offset*cell_size_x + box1_x_norm*cell_size_x 
        box1_y = y_offset*cell_size_y + box1_y_norm*cell_size_y 
        
        box1_w = image_width  * box1_w_norm 
        box1_h = image_height * box1_h_norm 
        
        box1_top_left_coor_x = box1_x - box1_w/2
        box1_top_left_coor_y = box1_y - box1_h/2
        box1_btm_rght_coor_x = box1_x + box1_w/2
        box1_btm_rght_coor_y = box1_y + box1_h/2
    
        '''
        bounding box 2 coordinate
        '''
        
        box2_x_norm,box2_y_norm = box2[:,i,0], box2[:,i,1] 
        box2_w_norm,box2_h_norm = box2[:,i,2], box2[:,i,3]
        
        box2_x = x_offset*cell_size_x + box2_x_norm*cell_size_x 
        box2_y = y_offset*cell_size_y + box2_y_norm*cell_size_y 
        
        box2_w = image_width  * box2_w_norm 
        box2_h = image_height * box2_h_norm 
        
        box2_top_left_coor_x = box2_x - box1_w/2 
        box2_top_left_coor_y = box2_y - box2_h/2
        box2_btm_rght_coor_x = box2_x + box1_w/2
        box2_btm_rght_coor_y = box2_y + box2_h/2
        
        '''
        Calculate the IoU between each predicted box and the ground truth box
        '''
        
        #ground truth box area. 
        #We need to add 1 because the coordinate starts from 0, hence the ending coordinate will always be 1 
        #unit less than the image size. E.g. For an image of size 500 x 500. The starting coordinate is (0,0) 
        #while the ending coordinate is (499,499).
        truth_box_area = (truth_btm_rght_coor_x - truth_top_left_coor_x + 1) * (truth_btm_rght_coor_y - 
                                                                                truth_top_left_coor_y + 1)
        
        '''
        Box 1 and the ground truth
        '''
        #determine the x and y coordinates of the intersection rectangle
        box1_x1 = tf.maximum(box1_top_left_coor_x, truth_top_left_coor_x)
        box1_y1 = tf.maximum(box1_top_left_coor_y, truth_top_left_coor_y)
        box1_x2 = tf.minimum(box1_btm_rght_coor_x, truth_btm_rght_coor_x)
        box1_y2 = tf.minimum(box1_btm_rght_coor_y, truth_btm_rght_coor_y)
        
        #if the difference is less than 0, it means the boxes does not intersect
        overlap_area = tf.maximum(0.0, box1_x2 - box1_x1) * tf.maximum(0.0, box1_y2 - box1_y1)
        
        #area of the first bounding box
        box1_area = (box1_btm_rght_coor_x - box1_top_left_coor_x + 1) * (box1_btm_rght_coor_y - 
                                                                         box1_top_left_coor_y + 1)
        
        box1_iou = overlap_area/(box1_area + truth_box_area - overlap_area)
        
        '''
        Box 2 and the ground truth
        '''
        #determine the x and y coordinates of the intersection rectangle
        box2_x1 = tf.maximum(box2_top_left_coor_x, truth_top_left_coor_x)
        box2_y1 = tf.maximum(box2_top_left_coor_y, truth_top_left_coor_y)
        box2_x2 = tf.minimum(box2_btm_rght_coor_x, truth_btm_rght_coor_x)
        box2_y2 = tf.minimum(box2_btm_rght_coor_y, truth_btm_rght_coor_y)
        
        #if the difference is less than 0, it means the boxes does not intersect
        overlap_area = tf.maximum(0.0, box2_x2 - box2_x1) * tf.maximum(0.0, box2_y2 - box2_y1)
        
        #area of the first bounding box
        box2_area = (box2_btm_rght_coor_x - box2_top_left_coor_x + 1) * (box2_btm_rght_coor_y - 
                                                                         box2_top_left_coor_y + 1)
        
        box2_iou = overlap_area/(box2_area + truth_box_area - overlap_area + 1e-9)
        
        IoU_list.append([box1_iou, box2_iou])
    
    #tensor of the IoU in shape of [batch_size, 49, 2]
    IoU_array = tf.reshape(tf.convert_to_tensor(IoU_list), (-1, 49,2))
    
    
    first_box_iou  = IoU_array[:,:,0] #shape : [batch_size, 49]
    second_box_iou = IoU_array[:,:,1] #shape : [batch_size, 49]
    
    #returns a tensor of size [batch_size, 49] with 1.0 and 0.0 in its elements
    #bb_select_box1 holds the truth values if box1's IoU is bigger than box2 or not. 1.0 for True ,0.0 for False
    #bb_select_box2 holds the truth values if box2's IoU is bigger than box1 or not. 1.0 for True ,0.0 for False
    bb_select_box1 = tf.cast(tf.greater(first_box_iou, second_box_iou), tf.float32)
    bb_select_box2 = tf.cast(tf.greater(second_box_iou, first_box_iou), tf.float32)
    
    return (bb_select_box1, bb_select_box2, IoU_array)


# In[11]:



X       = tf.placeholder(tf.float32, shape=(None, image_height, image_width, image_depth), name='X') 
                                                                                    #(batch_size, 448, 448, 3)
Y       = tf.placeholder(tf.float32, shape=(None, S**2, 5+C), name='Y') #(batch_size, 49, 25)
dropout = tf.placeholder(tf.float32, name='dropout') #dropout rate

#output size : (batch_size, 224, 224, 64)
conv1 = tf.contrib.layers.conv2d(X, num_outputs=64, kernel_size=7, stride=2, 
                                 padding='SAME', activation_fn=tf.nn.leaky_relu)

#output size : (batch_size, 112, 112, 64)
conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#output size : (batch_size, 112, 112, 128)
conv2 = tf.contrib.layers.conv2d(conv1_pool, num_outputs=128, kernel_size=3, stride=1, 
                                 padding='SAME', activation_fn=tf.nn.leaky_relu)

#output size : (batch_size, 56, 56, 128)
conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#output size : (batch_size, 56, 56, 192)
conv3 = tf.contrib.layers.conv2d(conv2_pool, num_outputs=192, kernel_size=1, stride=1,
                                padding='SAME', activation_fn=tf.nn.leaky_relu)

#output size : (batch_size, 56, 56, 256)
conv4 = tf.contrib.layers.conv2d(conv3, num_outputs=256, kernel_size=3, stride=1,
                                padding='SAME', activation_fn=tf.nn.leaky_relu)

#output size : (batch_size, 56, 56, 256)
conv5 = tf.contrib.layers.conv2d(conv4, num_outputs=256, kernel_size=1, stride=1,
                                padding='SAME', activation_fn=tf.nn.leaky_relu)

#output size : (batch_size, 28, 28, 256)
conv5_pool = tf.nn.max_pool(conv5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#output size : (batch_size, 28, 28, 512)
conv6 = tf.contrib.layers.conv2d(conv5_pool, num_outputs=512, kernel_size=3, stride=1,
                                padding='SAME', activation_fn=tf.nn.leaky_relu)

#output size : (batch_size, 28, 28, 512)
conv7 = tf.contrib.layers.conv2d(conv6, num_outputs=512, kernel_size=1, stride=1,
                                padding='SAME', activation_fn=tf.nn.leaky_relu)

#output size : (batch_size, 14, 14, 512)
conv7_pool = tf.nn.max_pool(conv7, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#output size : (batch_size, 14, 14, 600)
conv8 = tf.contrib.layers.conv2d(conv7_pool, num_outputs=600, kernel_size=3, stride=1,
                                padding='SAME', activation_fn=tf.nn.leaky_relu)

#output size : (batch_size, 7, 7, 600)
conv8_pool = tf.nn.max_pool(conv8, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

#output size : (batch_size, 7, 7, 600)
final_conv = tf.contrib.layers.conv2d(conv8_pool, num_outputs=600, kernel_size=3, stride=1,
                                padding='SAME', activation_fn=tf.nn.leaky_relu)

output_shape = 7*7*600
#feature vector shape : (batch_size, 29400)
feature_vector = tf.reshape(final_conv, (-1, 7*7*600))

#Weight and bias variables for Fully connected layers
W1 = tf.Variable(tf.truncated_normal([output_shape, 4096], stddev=0.1))
B1 = tf.Variable(tf.constant(1.0, shape=[4096]))
W2 = tf.Variable(tf.truncated_normal([4096, 7*7*30], stddev=0.1))
B2 = tf.Variable(tf.constant(1.0, shape=[7*7*30]))

#First fully-connected layer
fc1 = tf.add(tf.matmul(feature_vector, W1), B1)
fc1_actv = tf.nn.leaky_relu(fc1) #non-linear actv func

#dropout
dropout_layer = tf.nn.dropout(fc1_actv, dropout)

#Second fully-connected layer
fc2 = tf.add(tf.matmul(dropout_layer, W2), B2)

Y_pred = tf.nn.sigmoid(fc2) #shape : [batch_size, 7*7*30]             


# In[12]:


#Loss function



prediction = tf.reshape(Y_pred, (-1, 49, 30), name='prediction')

#input first bounding box, second bounding box and the ground truth bounding box
box_selection1, box_selection2, iou = bbox_selector(prediction[:,:,0:4], prediction[:,:,5:9], Y[:,:,:4])

#box_selection will ensure to pick the highest IoU predicted bounding box, while Y[:,:,4] at the beginning of
#the term will ensure if there's any object at all in a particular cell.
loss_1 = lambda_coord *(tf.reduce_sum(Y[:,:,4] * (box_selection1 *
                                                 ((prediction[:,:,0] - Y[:,:,0])**2 + 
                                                  (prediction[:,:,1] - Y[:,:,1])**2) +
                                                  box_selection2 * 
                                                  ((prediction[:,:,5] - Y[:,:,0])**2 +
                                                   (prediction[:,:,6] - Y[:,:,1])**2))))


loss_2 = lambda_coord *(tf.reduce_sum(Y[:,:,4] * (box_selection1 *
                                                 ((tf.sqrt(prediction[:,:,2]+ epsilon) - tf.sqrt(Y[:,:,2]+ epsilon) )**2 + 
                                                  (tf.sqrt(prediction[:,:,3]+ epsilon)- tf.sqrt(Y[:,:,3]+ epsilon))**2) +
                                                  box_selection2 * 
                                                  ((tf.sqrt(prediction[:,:,7]+ epsilon) - tf.sqrt(Y[:,:,2]+ epsilon))**2 +
                                                   (tf.sqrt(prediction[:,:,8]+ epsilon) - tf.sqrt(Y[:,:,3]+ epsilon))**2))))

#this part, I'm not sure if I understood it correctly
loss_3 = tf.reduce_sum(Y[:,:,4] * (box_selection1 * 
                                   (prediction[:,:,4] * iou[:,:,0]) +
                                    box_selection2* 
                                   (prediction[:,:,9] * iou[:,:,1]) -
                                    (Y[:,:,4] *(box_selection1*iou[:,:,0] + box_selection2*iou[:,:,1])))**2)

#change the 1.0 into 0.0 and 0.0 into 1.0 in Y[:,:,4].
#cast the float to bool and back to float since logical_not module requires bool type data
logical_not = tf.cast(tf.logical_not(tf.cast(Y[:,:,4], tf.bool)), tf.float32)

loss_4 = lambda_noobj * (tf.reduce_sum(logical_not * (box_selection1 * 
                                   (prediction[:,:,4] * iou[:,:,0]) +
                                    box_selection2* 
                                   (prediction[:,:,9] * iou[:,:,1]) -
                                    (Y[:,:,4] *(box_selection1*iou[:,:,0] + box_selection2*iou[:,:,1])))**2))

loss_5 = tf.reduce_sum(Y[:,:,4] * tf.reduce_sum((prediction[:,:,11:] - Y[:,:,6:])**2, axis=2))

total_loss = loss_1 + loss_2+ loss_3 + loss_4 + loss_5

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

tf.add_to_collection('prediction', prediction)
tf.add_to_collection('X', X)
tf.add_to_collection('Y', Y)
tf.add_to_collection('dropout', dropout)


                            
# second_part_loss = lambda_coord * tf.reduce_sum(Y[:,:,4] * ((tf.sqrt(prediction[:,:,2]) - tf.sqrt(Y[:,:,2]))**2 
#                                               + (tf.sqrt(prediction[:,:,3]) - tf.sqrt(Y[:,:,3])**2)))


# In[13]:


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())

#exception handling in loading the model
try:
	#restore the saved model
	saver.restore(sess, model_save_path)
	print("Model has been loaded !")
except:
	print("Model is not loaded !")

total_images = len(list_images)

for epoch_idx in range(epoch):
    
    loss = 0
    
    for i in range(0, total_images, batch_size):

        end_batch_size = i + batch_size

        if end_batch_size >= total_images : 

            end_batch_size = total_images - 1

        images, labels = load_dataset(i, end_batch_size)

        loss += sess.run([total_loss, optimizer], feed_dict={X:images, Y:labels, dropout:1.0})[0]
        
    print("Epoch : ", str(epoch_idx), "Loss : ", str(loss))
    #record loss log
    file = open('loss_record.txt', 'a')
    file.write("The loss at epoch " + str(epoch_idx) + " is : " + str(loss) + " \n")
    file.close()

    saver.save(sess, model_save_path)

#save model



