import numpy as np
import cv2
import tensorflow as tf

model = tf.keras.models.load_model("CNN_Model.h5")

def img_class(model, img):
    img_arr = np.asarray(img)
    
    pred_probab = model.predict(img_arr)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    return max(pred_probab), pred_class


# Initializing Video Frame
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    # Our operations on the frame come here
    # Displaying border in frame
    frame = cv2.rectangle(frame, (40,100), (240,300), (0,255,0), thickness = 1)
    
    # Cropping Hand Region
    crop = frame[100:300, 40:240]
    
    # Converting to GRAY
    img_gry = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # Applying Gaussian BLur                       
    img_gry_blr = cv2.GaussianBlur(img_gry, (5,5), 0)
    
    # Resizing
    img_1 = cv2.resize(img_gry_blr, (28,28), interpolation = cv2.INTER_AREA)
    img_2 = np.resize(img_1, (28,28,1))
    img_3 = np.expand_dims(img_2, axis = 0)
    
        
    pred_probab, pred_class = img_class(model, img_3)
    
    char_op = chr(pred_class + 65)

    cv2.putText(frame, char_op, (580,420), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    cv2.imshow('gry',img_gry)
    cv2.imshow('blur',img_gry_blr)
    
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()