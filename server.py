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
from src.predictor import PoPredictor
from src.utils import parse_args

app = Flask(__name__)


@app.route("/predict", methods=["post"])
def predict():
    req = request.json
    resp = model.predict(req["text"], task="ee", \
                        paragraphing=req.get("paragraphing", True), \
                        require_doccano_label=req.get("require_doccano_label", False))
    return jsonify(resp)


if __name__ == "__main__":
    
    config = parse_args()
    print(config)

    model = PoPredictor(config.model_name_or_path, \
                    config.extra_model_name_or_path, \
                    model_type=config.model_type, \
                    inference_mode=config.inference_mode, \
                    pre_seq_len=config.pre_seq_len, \
                    quantize=config.quantize)
    app.run(host=config.host, port=config.port)