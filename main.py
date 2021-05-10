#Import necessary libraries
from flask import Flask, render_template, Response
from camera import VideoCamera

#Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
        return Response(VideoCamera().gen_frame(), 
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == "__main__":
    app.run()