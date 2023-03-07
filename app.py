from flask import *
import sys
import os
import io
from stat import *
#import fastai
#from fastai import *
#from fastai.vision import *
#from fastai.imports import *
from fastai.vision.all import *
import base64
from flask_cors import CORS, cross_origin

import pathlib
#temp = pathlib.PosixPath
#pathlib.PosixPath = pathlib.WindowsPath

# Khởi tạo server
app = Flask(__name__)
CORS(app)
# Khai báo đường dẫn thư mục
current_directory = os.getcwd()
path = Path(current_directory)

# Load model
learner = load_learner(path/'export.pkl')

# Khai báo hàm xử lý request detect
@app.route('/detect', methods=['POST'])
@cross_origin()
def predict_img():
    requestData = json.loads(request.data.decode('utf-8'))
    # Lấy dữ liệu image dạng base64
    # b64 = request.form.get('image')
    b64 = requestData['image']
    # Chuyển về dạng bytes
    b64 = bytes(b64, 'utf-8')
    # Chuyển thành ảnh imageToSave.png
    with open("imageToSave.png", "wb") as fh:
        fh.write(base64.decodebytes(b64))
    # Đọc ảnh từ imageToSave.png và phân loại ô nhiễm
    image = 'imageToSave.png'
    pred_class, pred_idx, outputs = learner.predict(image)
    # Trả về dạng json của các loại ô nhiễm
    result = {'nomal':round(float(outputs[1])*100,4), 'air_pollution':round(float(outputs[0])*100,4), 'soil_pollution':round(float(outputs[2])*100,4), 'water_pollution':round(float(outputs[3])*100,4)}
    return result

# Thực thi server
if __name__ == "__main__":
    app.run()
    #  app.run(debug=False, host='0.0.0.0',port=8000)