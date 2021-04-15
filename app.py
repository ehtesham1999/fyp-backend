from flask import Flask
from electronics.electronics import electronics
from sports.sports import sports
from cellphone.cellphone import cellphone
from flask_cors import CORS
app = Flask(__name__)
app.register_blueprint(electronics,url_prefix="/electronics")
app.register_blueprint(sports,url_prefix="/sports")
app.register_blueprint(cellphone,url_prefix="/cellphone")

CORS(app)
nb_closest_images = 5


@app.route('/', methods=['GET'])
def hello():
    return "<h1>Welcome</h1>"

if __name__ == '__main__':
    app.run()
