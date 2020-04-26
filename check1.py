import cv2
import numpy as np
import time

min_confidence = 0.3
margin = 30
#file_name = "drone3.jpg"

# Load Yolo
net = cv2.dnn.readNet("yolo/yolo-drone.weights", "yolo/yolo-drone.cfg")
classes = []
with open("yolo/drone.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
print(classes)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0,255,size=(len(classes),3))
# Loading image
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0

while True:
    _,frame = cap.read()
    
    height,width,channels = frame.shape
    
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # Filter only 'drone'
            if class_id == 0 and confidence > min_confidence:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    color = (0, 255, 0)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = '{:,.2%}'.format(confidences[i])
            print(i, label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)      #making box
            cv2.putText(frame, label, (x, y - 10), font, 1, color, 2)   #apear %
            
            '''
            if len(indexes) != 0:
                most = label.index(max(label))
                print('most',most)
            
                lx,ly,lw,lh = boxes[most]
                x_medium = int((lx+lw)/2)
                y_medium = int((ly+lh)/2)
                cv2.line(frame, (x_medium, 0), (x_medium, 600), (0, 255, 0), 2)
                cv2.line(frame, (0, y_medium), (800, y_medium), (0, 255, 0), 2)    #sd
            
              '''  
            
            
            
            
    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2, (0,0,0),1)
        
    text = "Number of drone is : {} ".format(len(indexes))
    cv2.putText(frame, text, (margin, margin), font, 2, color, 2)

    cv2.imshow("Number of done ", frame)

    '''
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))
    '''
    key = cv2.waitKey(1)
    
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()


