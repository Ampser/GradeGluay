from flask import Flask
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from gradegluay.config import Config


limiter = Limiter(key_func=get_remote_address)


def create_app(config_class=Config):
    from gradegluay.routes import main_bp

    app = Flask(
        __name__,
        static_folder="../static",
        template_folder="../templates",
    )
    app.config.from_object(config_class)

    app.config["UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["ANNOTATED_UPLOAD_FOLDER"].mkdir(parents=True, exist_ok=True)
    app.config["DATA_DIR"].mkdir(parents=True, exist_ok=True)

    limiter.init_app(app)
    app.register_blueprint(main_bp)

    return app
