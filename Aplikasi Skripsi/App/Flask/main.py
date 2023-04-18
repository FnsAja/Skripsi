import sys
import os
import json
import pandas as pd
sys.path.append("D:\\Alfonso's Personal File\\Kuliah\\Skripsi\\Aplikasi Skripsi\\Function")

from flask import Flask, render_template, request, flash, send_file, redirect, url_for
from tweet import insertToExcel
from preprocessing import startTrain
from predict import startPredict, startPredictt

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'
pesan = {}

@app.route('/')
def redir():
    return redirect(url_for('scrap'))

@app.route('/scrap', methods=('GET', 'POST'))
def scrap():
    if request.method == 'POST':
        name = request.form['name']
        start = request.form['start']
        end = request.form['end']

        if name == '' or start == None or end == None:
            flash('Pilihan tidak boleh kosong')
        else:
            insertToExcel(name, start, end)
            return render_template('get_data.html', status = 'complete', name = name)
        
    return render_template('get_data.html', pesan = pesan)

@app.route('/train', methods=('GET', 'POST'))
def train():
    status = ''
    if request.method == 'POST':
        file = request.files['file'] 

        if file.filename == '':
            flash('File Tidak Boleh Kosong')
        elif file.filename.split('.')[-1] != 'xlsx':
            print(file.filename.split('.')[-1])
            flash('Ekstensi File Tidak Sesuai')
        else:
            if not os.path.exists('TrainData'):
                os.mkdir('TrainData')

            file.save('TrainData/Train.xlsx')

            status = 'complete'

            best_fold, all_fold, df_value = startTrain('TrainData/Train.xlsx')
            best_fold.pop("clf")
            best_fold.pop("x_test")
            best_fold.pop("y_test")

            return render_template('train.html', status = status, best_fold = json.dumps(best_fold), all_fold = json.dumps(all_fold), df_value = df_value)
        
    return render_template('train.html', status = status)

@app.route('/predict', methods=('GET', 'POST'))
def predict_file():
    status = ''
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            flash('File Tidak Boleh Kosong')
        elif file.filename.split('.')[-1] != 'xlsx':
            print(file.filename.split('.')[-1])
            flash('Ekstensi File Tidak Sesuai')
        else:
            if not os.path.exists('TestData'):
                os.mkdir('TestData')
                
            file.save('TestData/Test.xlsx')
            
            status = 'complete'
            
            positiveWords, netralWords, negativeWords, countPositive, countNetral, countNegative = startPredict('Model/svm.pkl', 'TestData/Test.xlsx')
            
            print(positiveWords, negativeWords, netralWords)
            print(countNegative, countNetral, countPositive)

            return render_template('predict.html', status = status, data = json.dumps({'positiveWords': positiveWords, 'netralWords': netralWords, 'negativeWords': negativeWords, 'countPositive': countPositive, 'countNetral': countNetral, 'countNegative': countNegative}))
    
    return render_template('predict.html', status = status)

@app.route('/predictt', methods=['POST'])
def predict_text():
    status = ''
    if request.method == 'POST':
        text = request.form['text']

        if text == '':
            flash('Please Input Something')
            return render_template('predict.html', status = status)
        else:
            status = 'complete'
            
            processed_features, result = startPredictt('Model/svm.pkl', text)
            
            return render_template('predict.html', status = status, data = {'processed_features': processed_features, 'result': result})
    
    return render_template('predict.html', status = status)

@app.route('/saveFile/<name>')
def save_excel(name):
    return send_file(
        f'Data/output_{name}.xlsx',
        download_name='Data.xlsx',
        as_attachment=True
    )
    
@app.route('/saveFileTest')
def save_excel_test():
    return send_file(
        f'Process/DataTest.xlsx',
        download_name='DataTest.xlsx',
        as_attachment=True
    )
    
@app.route('/saveFilePredict')
def save_excel_predict():
    return send_file(
        f'Process/DataPredict.xlsx',
        download_name='DataPredict.xlsx',
        as_attachment=True
    )

@app.route('/trainPlot')
def save_trainPlot_image():
    filename = './TrainData/TrainPlot.png'
    return send_file(filename, mimetype='image/png')

@app.route('/trainChart')
def save_trainChart_image():
    filename = './TrainData/TrainChart.png'
    return send_file(filename, mimetype='image/png')

@app.route('/trainWordcloud<name>')
def save_wordcloud_image(name):
    filename = f'./TrainData/{name}TrainWordCloud.png'
    return send_file(filename, mimetype='image/png')

@app.route('/predictWordcloud<name>')
def save_predwordcloud_image(name):
    filename = f'./TrainData/{name}TrainWordCloud.png'
    return send_file(filename, mimetype='image/png')