import flask
from pathlib import Path
from flask import Flask, request, render_template
import matplotlib.pyplot as plt
import joblib
import numpy as np
from scipy import misc
from librosa import display
from ml.model import export_model
from flask_restful import Resource, Api
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
from PIL import Image
import wave
import librosa

app = Flask(__name__)
api = Api(app)

dir = str(Path(__file__).parent)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')

@app.route("/record")
def record():
    return flask.render_template('record.html')

# 데이터 예측 처리
@app.route('/predict', methods=['POST'])
def make_prediction():
    TARGET_SIZE=224
    BATCH_SIZE = 8
    if request.method == 'POST':
        file = request.files['audio']

        if not file:
            return render_template('index.html', ml_label="No Files")
        # wav = wave.open(file,mode='rb')
        wav, sr = librosa.load(file)
        frame_length = 0.025
        frame_stride = 0.010
        input_nfft = int(round(sr * frame_length))
        input_stride = int(round(sr * frame_stride))
        S = librosa.feature.melspectrogram(wav, sr=sr)
        S_DB = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=input_stride)
        image=dir+"/test/some/test.png"
        plt.savefig(image)
        plt.close()

        td = dir+"/test"
        tds = image_dataset_from_directory(
            td,
            seed=123,
            image_size=(TARGET_SIZE, TARGET_SIZE),
            batch_size=BATCH_SIZE)

        pre = model.predict(tds)
        target =0
        l =0
        os.remove(dir + "/test/some/test.png")
        for index,num in enumerate(pre[0]):
            if target<num:
                l=index
                target=num
        if l==0:
            label ="blues"
        elif l==1:
            label = "classical"
        elif l==2:
            label = "country"
        elif l==3:
            label = "disco"
        elif l==4:
            label = "hiphop"
        elif l==5:
            label = "jazz"
        elif l==6:
            label = "metal"
        elif l==7:
            label = "pop"
        elif l==8:
            label = "reggae"
        elif l==9:
            label = "rock"
        # 결과 리턴
        return render_template('index.html', ml_label=label)



@app.route('/record')
def record_music():
    return render_template('record.html')


@app.route('/addData', methods=['POST'])
def add_data():
    if request.method == 'POST':
        file2 = request.files['adddata']
        genre= request.form.get("genre")
        if not file2:
            return render_template('index.html', ml_label="No Files")
        # print(file2)
        # print(genre)
        wav, sr = librosa.load(file2)
        frame_length = 0.025
        frame_stride = 0.010
        input_nfft = int(round(sr * frame_length))
        input_stride = int(round(sr * frame_stride))
        S = librosa.feature.melspectrogram(wav, sr=sr)
        S_DB = librosa.amplitude_to_db(S, ref=np.max)
        librosa.display.specshow(S_DB, sr=sr, hop_length=input_stride)
        image=dir+"/data/images_original/"+genre+"/test.png"
        plt.savefig(image)
        plt.close()
        return render_template('index.html', ml_label="加载数据完成")





# 数据重新训练
@app.route('/retrain', methods=['POST'])
def make_model():
    if request.method == 'POST':
        # 模型重新生成
        export_model('R')
        return render_template('index.html', md_label='重新学习完成')


# 数据重新训练(RestApi)
class RestMl(Resource):
    def get(self):
        export_model('R')
        return {'result': True, 'modelName': 'model.pkl'}


# Rest 登录
api.add_resource(RestMl, '/retrainModel')

if __name__ == '__main__':

    # 모델 로드
    # ml/model.py 先实行后生成
    model = tf.keras.models.load_model(dir+"/model/model.h5")
    # Flask 服务器开始
    app.run(host='localhost', port=8000, debug=True)
