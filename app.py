# app.py
import os

from flask import Flask
from routes import register_blueprints

app = Flask(__name__)
# Register blueprints
register_blueprints(app)
# register_sockets(socketio)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
