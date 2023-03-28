import sys
import os
sys.path.append("D:\\Alfonso's Personal File\\Kuliah\\Skripsi\\Aplikasi Skripsi\\Function")

from flask import Flask, render_template, request, flash, send_file, redirect, url_for
from tweet import insertToExcel
from preprocessing import startTrain

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
            pesan['name'] = name
            pesan['start'] = start
            pesan['end'] = end
            
            insertToExcel(name, start, end)
            return redirect(url_for('save_excel', name = name))
        
    return render_template('get_data.html', pesan = pesan)

@app.route('/saveFile/<name>')
def save_excel(name):
    return send_file(
        f'Data/Data_{name}.zip',
        mimetype='application/zip',
        download_name='Data.zip',
        as_attachment=True
    )

@app.route('/train', methods=('GET', 'POST'))
def train():
    if request.method == 'POST':
        file = request.files['file']

        if file.filename == '':
            flash('Please Input File')
        else:
            if not os.path.exists('TrainData'):
                os.mkdir('TrainData')

            file.save('TrainData/Train.xlsx')

            startTrain('TrainData/Train.xlsx')
    
    return render_template('train.html')