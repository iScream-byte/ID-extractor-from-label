ID extractor from label
overview: Extracting the ID of an attached label on a product bag and converting the ID to text.

There will be a label attached to a bag  of a product. The main requirement is to scan the label using a camera and detect it’s ID region. Then the detected ID region is cropped from the main captured image of the label. The detected ID is then passed  through several process to detect it’s contents and the image is tranferred  into text. The text could be used for further processes.
Step 1: Capturing images for the training dataset.
Using open cv, an array of labels are  captured.
Step 2: Then using labelIMG, the ID region is labelled and the region is saved as an XML file along with the main image file. 
Step 3: A transfer learning is initiated using the labelled image along with the XML files. A pretrained model i.e. my_ssd_mobnet is downloaded and used for the training. 
Step 4: After training phase, the model can successfully detect the ID region of the image. The model also returns the bounding box of the detected region.
Step 5:  Using the bounding box co ordinates, the ID region of the label is the cropped and saved to storage. Then the OCR portion of the process is initiated. 


Training the OCR: 
Step 1: Generating the dataset. Images of arrays of handwritten letters are captured. 
Step 2: The image is then enhanced  and every single letter is extracted from the main image by sorting the contours, added a little padding to every image and saved.
Step 3: The extracted images are grouped manually and stored in folders.
Step 4: A VGG16 replica model is built using tensorflow for the OCR. The model is trained using  these images. The model is trained to predict the class name ie the folder name of an inputted image. 
Step 5: After training the model, the model is saved.
Step 6: After completion of step 5 of the first section, the ID region is then passed to some function and each letter of that region is identified and distinguished using a bounding box.
Step 7: Each image of the letters of the ID region is collected and fed  into the model to be predicted its class.
Step 8: An empty list is created and the predictions of the model is appended to that empty list. Transferring the list into string produces the true output ie the  string that is written in the detection region.
