import json

import requests
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from tablut import Tafl

app = Flask(__name__, static_url_path='', static_folder='web/static')
app.config['SECRET_KEY'] = 'some-super-secret-key'
app.config['DEFAULT_PARSERS'] = [
    'flask.ext.api.parsers.JSONParser',
    'flask.ext.api.parsers.URLEncodedParser',
    'flask.ext.api.parsers.FormParser',
    'flask.ext.api.parsers.MultiPartParser'
]
app.debug = True
cors = CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app)
alive_objects = {}

@app.route('/')
def index():
    return render_template('board.html')


@socketio.on('connect', namespace="/tablut")
def start_board():
    game = Tafl()
    g_json = game.json()
    alive_objects[hex(id(game))] = game
    emit('start_game', {'data': g_json})


@socketio.on('move', namespace='/tablut')
def player_move(data):
    
    #board, stats, step = msg['state'], msg['stats'], msg['move']
    game = Tafl(fromjson=json.dumps(data['state']))
    game.in_step(int(data['movement']))
    print(game)
    emit('move', {'data':game.json()})


# @app.after_request
# def add_header(response):
#     response.headers['Access-Control-Allow-Origin'] = '*'
#     response.headers['Access-Control-Allow-Headers'] = 'Access-Control-Allow-Headers, Origin, X-Requested-With, Content-Type, Accept, Authorization'
#     response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, HEAD'
#     return response


# @app.route('/', defaults={'path': ''})
# @app.route('/<path:path>')
# def catch_all(path):
#     if app.debug:
#         return requests.get('http://localhost:3000/{}'.format(path)).text
#     return render_template("index.html")


if __name__ == '__main__':
    socketio.run(app)
