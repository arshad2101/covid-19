from flask import Flask, url_for, send_from_directory, request,render_template
import logging, os, json
from werkzeug import secure_filename
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)
app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/test/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# Imports
import keras
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import load_img
from IPython.display import display
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

test_batchsize = 10   
image_size = 256      
test_dir = 'uploads'
#train_dir = 'resized_train'
    
def create_new_folder(local_dir):
    newpath = local_dir
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    return newpath

@app.route('/postjson', methods=['POST'])
def post():
    
    print(request.is_json)
    content = request.get_json()
    #print(content)
    print(content['id'])
    print(content['name'])
    return 'JSON posted'
    
@app.route('/upload', methods = ['POST'])
def api_root():
    
    app.logger.info(PROJECT_HOME)
    if request.method == 'POST' and request.files['image']:
    	#print(request.is_json)
    	#content = request.get_json()
    	print(request)
    	params = request.values
    	names=params.get('name')
    	ids=params.get('id')
    	app.logger.info(app.config['UPLOAD_FOLDER'])
    	img = request.files['image']
    	img_name = secure_filename(img.filename)
    	create_new_folder(app.config['UPLOAD_FOLDER'])
    	saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
    	app.logger.info("saving {}".format(saved_path))
    	img.save(saved_path)
    	predicted_classes = predict(img_name,names,ids)
    	#return send_from_directory(app.config['UPLOAD_FOLDER'],img_name, as_attachment=True)
    	return predicted_classes
    else:
    	return "Where is the image?"

import tensorflow as tf
session = tf.Session(graph=tf.Graph())
with session.graph.as_default():
    keras.backend.set_session(session)
    new_model = load_model('model.h5')
    



def test_datagenerator():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                              target_size=(image_size, image_size),
                              batch_size=test_batchsize,
                              class_mode='categorical',
                              shuffle=False)
 
    return test_dir, test_generator

# Train datagenerator
'''def train_datagenerator(test_batchsize):
    
    train_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow_from_directory(train_dir)

    return train_generator'''
    
def predict(img_name,names,ids):
    global sess
    global graph
   # print('loading trained model...')
    #new_model = keras.models.load_model('trained_models/model.h5')
   # print('loading complete')

   
    print('predicting on the test images...')
    test_dir1, test_generator = test_datagenerator()
    prediction_start = time.clock()            
    with session.graph.as_default():  
        keras.backend.set_session(session)    
        predictions = new_model.predict_generator(test_generator,
                                              steps=test_generator.samples / test_generator.batch_size,
                                              verbose=1)

    prediction_finish = time.clock()
    prediction_time = prediction_finish - prediction_start
    predicted_classes = np.argmax(predictions, axis=1)
    print(predictions)
    print(predicted_classes)
    filenames = test_generator.filenames
    #print(filenames)
    data = {}
    for (a, b, c) in zip(filenames, predicted_classes, predictions): 
        #print (a, b, c)
        data[a.split('\\')[len(a.split('\\'))-1]] = [a.split('\\')[len(a.split('\\'))-1],str(b),str(c)]  
        
    print(data)
    print (json.dumps(data, indent=4))
   
    
    
    #errors = np.where(predicted_classes != ground_truth)[0]
    #print("No. of errors = {}/{}".format(len(errors), test_generator.samples))

    #correct_predictions = np.where(predicted_classes == ground_truth)[0]
    #print("No. of correct predictions = {}/{}".format(len(correct_predictions), test_generator.samples))

    #print("Test Accuracy = {0:.2f}%".format(len(correct_predictions)*100/test_generator.samples))
    print("Predicted in {0:.3f} minutes!".format(prediction_time/60))
    #return json.dumps(data, indent=4)
    
    import base64
    base64img = ''
    with open('uploads/test/'+img_name,mode = 'rb') as image_file:
        img = image_file.read()
        base64img = base64.encodebytes(img).decode("utf-8")
    
    print(base64img)
    if b==0:
        d="covid"
    else:
        d="normal"
    listOfStr = ["Mild", "Moderate" , "NoDir" , "ProliferativeDR","Severe" ]
    zipbObj = zip(listOfStr, c)
    dictOfWords = dict(zipbObj)
    
    
    strhtml='<html><head><style>  body{background-color:RosyBrown;}#customers {font-family: "Lucida Console", Monaco, monospace;border-collapse: collapse;border-radius: 2em;overflow: hidden;width:80%;height:45%;margin-top: 150px;margin-right: 150px;margin-left:150px;}#customers td, #customers th {border: 1px solid #ddd;padding: 8px;}#customers tr:nth-child(even){background-color: LavenderBlush;}#customers tr:hover {background-color: #ddd;}#customers th {padding-top: 12px;padding-bottom: 12px;text-align: left;background-color: DeepSkyBlue;color: white;}</style></head><body><p></p><table id="customers" ><tr><th>PATIENT_NAME</th><th>PATIENT_ID</th><th>IMAGE_NAME</th><th>PREDICTED_CLASS</th><th>PREDICTIONS</th><th>IMAGE</th></tr><tr><td>'+names+'</td><td>'+ids+'</td><td>'+a[5:]+'</td><td>'+str(d)+'</td><td>'+str(dictOfWords)+'</td><td><img src="data:image/jpeg;base64,'+base64img+'"></td></tr></table></body></html>'

     

    return strhtml
    


    
'''train_generator = train_datagenerator(test_batchsize)
print('summary of loaded model')
new_model.summary()
ground_truth = train_generator.class_indices
print(ground_truth)'''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5001',debug=False)
