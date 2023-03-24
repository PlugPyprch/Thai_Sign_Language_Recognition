from fastapi import FastAPI, Request,Body
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

class_name = ['cold', 'dizziness', 'allergic_reaction',
'snot', 'unwell', 'hello', 'myself']

model1 = tf.keras.models.load_model('models/L64D32_9286.h5')
model2 = tf.keras.models.load_model('models/L64D64_7500.h5')
scaler = MinMaxScaler(feature_range=(0,1))

@app.post("/prediction")
async def prediction(payload: dict = Body(...)):
    data = payload['keypionts']
    arr = np.array(data)
    arr_data = scaler.fit_transform(arr.reshape(arr.shape[0], -1)).reshape(arr.shape)
    prediction = model1.predict(tf.expand_dims(arr_data, axis=0))
    result = class_name[np.argmax(prediction)]
    return result

# @app.post("/prediction")
# async def prediction(payload: dict = Body(...)):
#     # print(type(payload))
#     # print(payload.keys())
#     data = payload['keypionts']
#     arr = np.array(data)
#     arr_data = scaler.fit_transform(arr.reshape(arr.shape[0], -1)).reshape(arr.shape)
#     # prediction = model.predict(tf.expand_dims(arr_data, axis=0))
#     # result = class_name[np.argmax(prediction)]
#     return arr.shape