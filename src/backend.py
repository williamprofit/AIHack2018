from flask import Flask, render_template, redirect, url_for,request
from flask import make_response
from DataPreprocessor import DataPreprocessor
import csv
import os
from keras.models import load_model 
import numpy

app = Flask(__name__)

@app.route("/")
def home():
    return "hi"
@app.route("/index")

@app.route('/login', methods=['GET', 'POST'])
def login():
   message = None
   print("is called")
   if request.method == 'POST':
        print("is posted")
        datafromjs = request.form['mydata']
        print(datafromjs)
        vars = datafromjs.split(',')
        print(vars)
        vec_start = vars[0:-4]
        vec_start = results = [int(i) for i in vec_start]
        time = vars[-4]
        vec_end = vars[-3:-1]
        vec_end = results = [int(i) for i in vec_end]
        print("vec start: "+str(vec_start))
        print("vec end: "+str(vec_end))
        rel_path = vars[-1].strip()
        processor = DataPreprocessor()
        #outputs = processor.preprocess(processor, datafromjs)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, '../website/'+rel_path+'.csv')
        path = os.path.abspath(os.path.realpath(file_path))
        print(path)
        model_input = []
        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                vector = []
                row = [int(i) for i in row]
                vector += vec_start
                vector.append(time)
                vector += row
                vector += vec_end
                print("vector: " + str(len(vector)))
                model_input.append(vector)
        print("length: " + str(len(model_input)))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_path = os.path.join(dir_path, '../models/severity.h5')
        path = os.path.abspath(os.path.realpath(file_path))
        model = load_model(path)

        print("----")
        print(model_input[0])
        print("----")
        processor.preprocess(data=model_input)
        outputs = model.predict(processor.data)
        print(outputs)

        result = "return this"
        resp = make_response('{"response": '+result+'}')
        resp.headers['Content-Type'] = "application/json"
        return resp
        return render_template('login.html', message='')

if __name__ == "__main__":
    app.run(debug = True)
    
#get
'''
sex_of_driver, 
'''
#need
'''sex_of_driver,age_band_of_driver,age_of_vehicle,day_of_week,time,
road_type,speed_limit,junction_detail,
weather_conditions,vehicle_type'''