import os
import cv2
import PIL
import matplotlib.pyplot as plt
right = 25
left = 25
top = 25
bottom = 25

base_dir="D:\\iScream\\tfod\\Tensorflow\\training set backup\\training_set\\"
listdir=os.listdir(base_dir)

for i in listdir:
    current=base_dir+"\\"+i
    file_list=os.listdir(current)
    for j in file_list:
        image = PIL.Image.open(current+"\\"+j) 

        width, height = image.size
  
        new_width = width + right + left
        new_height = height + top + bottom
  
        result = PIL.Image.new(image.mode, (new_width, new_height), (255))  
        result.paste(image, (left, top))
        plt.imshow(result,cmap="gray")  
        result.save(current+"\\"+j)
