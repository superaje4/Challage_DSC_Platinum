import re
import pandas as pd
import sqlite3
from flask import Flask, jsonify, request, render_template, redirect, url_for
import pickle

app = Flask(__name__, template_folder='templates')

import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score


from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

dataset = pd.read_csv('news_article.csv', index_col=0)
train_data, test_data = train_test_split(dataset, test_size=20)
train_data, val_data = train_test_split(train_data, test_size=20)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from string import punctuation
stopwords_list = stopwords.words('indonesian')

def lowercasing(paragraph):

	return paragraph.lower()
	
def menghilangkan_tandabaca(paragraph):
	new_paragraph = re.sub(fr'[{punctuation}]', r'', paragraph)
	return new_paragraph

def text_normalization(paragraph):
	paragraph = lowercasing(paragraph)
	paragraph = re.sub(r"[ ]+",r' ',paragraph)
	return paragraph

lr = pickle.load(file=open("C:/Users/ASUS/DSC_binar/test api/pickle_file/logistic_regression.pkl",'rb'))
le = pickle.load(file=open("C:/Users/ASUS/DSC_binar/test api/pickle_file/label_encoder.pkl",'rb'))
cv = pickle.load(file=open("C:/Users/ASUS/DSC_binar/test api/pickle_file/count_vectorizer.pkl",'rb'))

      # <option value="Logistic Regression">Logistic Regression</option>
      # <option value="CNN">CNN</option>
      # <option value="NLP">NLP</option>
      # <option value="LSTM">LSTM</option>


#route untul logistic
@app.route('/', methods=['GET', "POST"])
def page_utama():
    if request.method == 'POST':
        opsi=request.form["pilihan"]
        if opsi=="Logistic Regression":
            return redirect(url_for("logistic"))
        elif opsi=="CNN":
            return redirect(url_for("CNN"))
        elif opsi=="LSTM":
            return redirect(url_for("LSTM"))
        elif opsi=="NLP":
            return redirect(url_for("NLP"))
    else:
        return render_template("page_utama.html")

#route untul logistic
@app.route('/logistic',methods=["GET","POST"])
def logistic():
    if request.method=="POST":
        opsi=request.form["tombol"]
        if opsi =="akurasi":
            return jsonify({"test":opsi})
        elif opsi =="prediksi":
            return redirect(url_for("logistic_prediksi"))
    else:
        return render_template("page_kedua.html")

@app.route('/logsitec_prediksi',methods=["GET","POST"])
def logistic_prediksi():
    if request.method=="POST":
        text=request.get["inputText"]
        pass
    else:
        return render_template("prediksi.html")

#route untuk CNN
@app.route('/CNN',methods=["GET","POST"])
def CNN():
    if request.method=="POST":
        opsi=request.form["tombol"]
        if opsi =="akurasi":
            return jsonify({"test":opsi})
        elif opsi =="prediksi":
            return redirect(url_for("CNN_prediksi"))
    else:
        return render_template("page_kedua.html")

@app.route('/CNN_prediksi',methods=["GET","POST"])
def CNN_prediksi():
    if request.method=="POST":
        text=request.get["inputText"]
        pass
    else:
        return render_template("prediksi.html")

#route untuk LSTM
@app.route('/LSTM',methods=["GET","POST"])
def LSTM():
    if request.method=="POST":
        opsi=request.form["tombol"]
        if opsi =="akurasi":
            return jsonify({"test":opsi})
        elif opsi =="prediksi":
            return redirect(url_for("LSTM_prediksi"))
    else:
        return render_template("page_kedua.html")

@app.route('/LSTM_prediksi',methods=["GET","POST"])
def LSTM_prediksi():
    if request.method=="POST":
        text=request.get["inputText"]
        pass
    else:
        return render_template("prediksi.html")


#route untuk NLP
@app.route('/NLP',methods=["GET","POST"])
def NLP():
    if request.method=="POST":
        opsi=request.form["tombol"]
        if opsi =="akurasi":
            return jsonify({"test":opsi})
        elif opsi =="prediksi":
            return redirect(url_for("NLP_prediksi"))
    else:
        return render_template("page_kedua.html")

@app.route('/NLP_prediksi',methods=["GET","POST"])
def NLP_prediksi():
    if request.method=="POST":
        text=request.get["inputText"]
        pass
    else:
        return render_template("prediksi.html")



if __name__ == '__main__':
    app.run(debug=True)

 