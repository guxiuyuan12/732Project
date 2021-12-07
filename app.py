from flask import Flask,render_template, request,send_from_directory
import webbrowser
from threading import Timer
from utility import prediction
from model import *
import os

model_vgg = VGGnetwork()
model_alex = AlexNet()
model_vgg.load_state_dict(torch.load('./saved_model/VGG_best_model.pth'))
model_alex.load_state_dict(torch.load('./saved_model/AlexNet_best_model.pth'))

app = Flask(__name__)
if(not os.path.exists('uploads')):
    print("uploads folder created!")
    os.mkdir('uploads')
app.config['UPLOAD_PATH'] = "uploads/"

@app.route("/", methods = ['GET'])
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
        res_vgg = prediction(model_vgg,file)
        res_alex = prediction(model_alex, file)
        return render_template('index.html',filename=filename,res_vgg=res_vgg,res_alex=res_alex)

@app.route('/uploads/<filename>')
def upload(filename):
    return send_from_directory(app.config['UPLOAD_PATH'], filename)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run(port=5000,debug=False)