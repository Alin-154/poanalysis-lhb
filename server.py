#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   server.py
@Time    :   2023/07/11 14:45:13
@Author  :   nicholas wu 
@Version :   1.0
@Contact :   nicholas_wu@aliyun.com
@License :    
@Desc    :   None
'''

from flask import Flask, request, jsonify

from .src.predictor import PoPredictor
from .src.utils import read_ini

app = Flask(__name__)

config = read_ini()

model = PoPredictor(config["model"]["model_name_or_path"], \
                    config["model"]["ptuning_checkpoint"], \
                    config["model"]["pre_seq_len"])


@app.route("/predict", methods=["post"])
def predict():
    req = request.json
    resp = model.predict(req["text"], task="ee")
    return jsonify(resp)


if __name__ == "__main__":
    app.run(host=config["server"]["host"], host=config["server"]["port"])