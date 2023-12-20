# app/routes/__init__.py
from .main import main_bp
from .predict import predict_bp


# Register blueprints
def register_blueprints(app):
    app.register_blueprint(main_bp)
    app.register_blueprint(predict_bp)


# def register_sockets(socket):
    # socket.on_event('message', handler=handle_messages)


# Import blueprints after defining them
from . import main, predict
