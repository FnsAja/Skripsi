from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')

@app.route('/')
def hello():
    return render_template('get_data.html')