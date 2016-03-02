 # -*- coding: utf-8 -*-
import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
import json

app = Flask(__name__, static_folder = 'assets')
@app.route('/assets/<path:path>')
def serve_static(path):
    root_dir = os.path.dirname(os.getcwd())
    return app.send_static_file(os.path.join(root_dir, 'assets', path))

@app.route('/', methods=['GET'])
def input():
    #input contains a simple form where we ask users whose stats they want to see 
    return render_template('input.html')