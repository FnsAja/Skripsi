import sys
sys.path.append("D:\\Alfonso's Personal File\\Kuliah\\Skripsi\\Aplikasi Skripsi\\Function")

from flask import Flask, render_template, request, flash, send_file, redirect, url_for
from tweet import insertToExcel

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'df0331cefc6c2b9a5d0208a726a5d1c0fd37324feba25506'
message = {}

@app.route('/', methods=('GET', 'POST'))
def hello():
    if request.method == 'POST':
        name = request.form['name']
        start = request.form['start']
        end = request.form['end']
        
        if name == '0':
            flash('choose any')
        else:
            message['name'] = name
            message['start'] = start
            message['end'] = end
            
            insertToExcel(name, start, end)
            return redirect(url_for('save_excel', name = name))
            # return render_template('get_data.html', message = message)
        
        
    return render_template('get_data.html')

@app.route('/saveFile/<name>')
def save_excel(name):
    return send_file(
        f'output_{name}.xlsx',
        mimetype='application/vnd.ms-excel',
        download_name='Data.xlsx',
        as_attachment=True
    )

@app.route('/<nama>/<mulai>/<selesai>')
def nama(nama, mulai, selesai):
    return nama + 'mulai : ' + mulai + 'selesai : ' + selesai