from flask import Flask

from app.routes import bp


app = Flask(__name__)

app.register_blueprint(bp)


app.route('static/<path:filename>')
def static_file(filename):
   return app.send_static_file(filename)

if __name__ == '__main__':
   app.run(debug = True)