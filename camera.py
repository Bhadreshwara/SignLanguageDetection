# Immporting the libraries
import numpy as np
import tensorflow as tf
import cv2

# Loading trained model
model = tf.keras.models.load_model('model1.h5')

# Using image as an array for prediction function
def recognize(img):
    img = np.resize(img, (28,28,1))
    img = np.expand_dims(img, axis=0)
    img = np.asarray(img)
    classes = model.predict(img)[0]
    pred_id = list(classes).index(max(classes))
    return pred_id


class VideoCamera(object):
    def __init__(self):
        # Initializing video frame
        self.cap = cv2.VideoCapture(0)
    
    def __del__(self):
        # Release the capture
        self.cap.release()
        
    def gen_frame(self):
        while True:
            # Capture frame-by-frame
            ret, frame = self.cap.read()   
            # Our operations on the frame come here
            # Displaying border for hand region
            frame = cv2.rectangle(frame, (40,100), (240,300), (0,255,0), 
                                  thickness = 1)
            # Cropping hand region part
            crop = frame[100:300, 40:240]
            # Converting to GRAY
            img_gry = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Applying Gaussian BLur                       
            img_gry_blr = cv2.GaussianBlur(img_gry, (5,5), 0)
            # Resizing
            img = cv2.resize(img_gry_blr, (28,28), interpolation=cv2.INTER_AREA)
            # Prediction
            y_pred = recognize(img)
            # Character equivalent
            char_op = chr(y_pred + 65)
            cv2.putText(frame, char_op, (580,420), cv2.FONT_HERSHEY_SIMPLEX, 2, 
                        (255,255,0), 2)
            # Display the resulting frame
            """cv2.imshow('frame', frame)
            cv2.imshow('gry_blr', img_gry_blr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break"""
            # Encoding raw frame to jpg
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
