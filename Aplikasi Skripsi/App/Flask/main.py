import sys
import os
import json
sys.path.append("D:\\Alfonso's Personal File\\Kuliah\\Skripsi\\Aplikasi Skripsi\\Function")

from flask import Flask, render_template, request, flash, send_file, redirect, url_for
from tweet import insertToExcel
from preprocessing import startTrain
from predict import startPredict

app = Flask(__name__, template_folder='templates')
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
            return redirect(url_for('save_excel', name = name))
        
    return render_template('get_data.html', pesan = pesan)

@app.route('/train', methods=('GET', 'POST'))
def train():
    status = ''
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            flash('Please Input File')
        else:
            if not os.path.exists('TrainData'):
                os.mkdir('TrainData')

            file.save('TrainData/Train.xlsx')

            status = 'complete'

            best_fold, all_fold = startTrain('TrainData/Train.xlsx')
            best_fold.pop("clf")

            return render_template('train.html', status = status, best_fold = json.dumps(best_fold), all_fold = json.dumps(all_fold))
        
    return render_template('train.html', status = status)

@app.route('/predict', methods=('GET', 'POST'))
def predict():
    status = ''
    if request.method == 'POST':
        file = request.files['file']
        
        if file.filename == '':
            flash('Please Input File')
        else:
            if not os.path.exists('TestData'):
                os.mkdir('TestData')
                
            file.save('TestData/Test.xlsx')
            
            status = 'complete'
            
            data = startPredict('Model/svm.pkl', 'TestData/Test.xlsx')
    
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
        f'Process/DataTest.csv',
        download_name='DataTest.csv',
        as_attachment=True
    )
    
@app.route('/saveFilePredict')
def save_excel_predict():
    return send_file(
        f'Process/DataPredict.xlsx',
        download_name='DataPredict.xlsx',
        as_attachment=True
    )