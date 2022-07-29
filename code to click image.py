import cv2
import uuid
import os


labels = ["G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","1","2","3","4","5","6","7","8","9","0"]
IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')
count=1

#!mkdir {IMAGES_PATH}
# for label in labels:
#     path = os.path.join(IMAGES_PATH, label)
#     if not os.path.exists(path):
#         !mkdir {path}
        


cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera")
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Camera", frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        #imgname = os.path.join(IMAGES_PATH,labels[0],labels[0]+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        imgname = os.path.join(IMAGES_PATH,labels[29],labels[29]+'.'+'{}.jpg'.format(str(count)))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(imgname, frame_gray)
        print("{} written!".format(imgname))
        count=count+1
cam.release()
cv2.destroyAllWindows()