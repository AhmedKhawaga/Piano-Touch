import cv2
import numpy as np
import glob
import line_detector

# Load Yolo
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["notes"]

# Images path
images_path = glob.glob("E:\\A7AAA\\KHWAGA\\notes\\*.jpeg")
#images_path.extend(glob.glob(r"D:\\done\\*.jpg"))

print(images_path)

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images

    # Loading image
img =cv2.imread("3.jpeg")
lines =  line_detector.lines(img)

#img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
centerX = []
centerY = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.9:
            # Object detected
            
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            centerX.append(center_x)
            centerY.append(center_y)
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = (255,0,0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color,2)
        cv2.circle(img,(centerX[i],centerY[i]),1,color,2)
        cv2.circle(img,(0,0),1,color,5)

centerX 
centerY
note =[''] *len(centerY)

for i in range(1,(len(lines)//5)+1):
    for j in range (len(centerY)) :
        
        if  centerY[j] - lines[0*i]>10 : 
            note[j] = 'do'
        
        elif 10 > centerY[j] - lines[0*i]>5 :
            note[j] = 're'
            
        elif  0 < centerY[j] - lines[0*i] < 10 or  10 > lines[0*i] - centerY[j] > 0 :
            note[j] ='mi'
            
        elif centerY[j] < lines[0*i]  and centerY[j] > lines[1*i] :
            if 0 < centerY[j] - lines[1*i]<10 or 10 > lines[1*i] - centerY[j]  > 0:
                note[j] ='sol'
            else:
                note[j] ='fa'
            
        elif centerY[j] < lines[1*i] and centerY[j] > lines[2*i]:
            note[j] ='la'
            
           
        elif centerY[j] < lines[2*i] and centerY[j] > lines[3*i]:
            if 0 < centerY[j] - lines[2*i]<10  or 10 > lines[2*i] - centerY[j] > 0:
                note[j] ='ti'
            else:
                note[j] ='do'
    
        elif centerY[j] < lines[3*i] and centerY[j] > lines[4*i] and (centerY[j] - lines[4*i]>5  or 5 < lines[4*i] - centerY[j]):
            if 0 < centerY[j] - lines[3*i]<10  or 10 > lines[3*i] - centerY[j] > 0:
                note[j] ='re'
            else:
                note[j] ='mi'
                
        elif centerY[j] > (lines[4*i]-30) :
            if 0 < centerY[j] - lines[4]*i<10  or 10 > lines[4*i] - centerY[j] > 0:
                note[j] ='fa'
            else:
                note[j] ='sol'
                
        else:
            note[j] = ' ' 
           
        cv2.putText(img,note[j],(centerX[j],centerY[j]+10),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        
cv2.imshow("Image", img)
key = cv2.waitKey(0)

cv2.destroyAllWindows()






