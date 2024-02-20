from flask import Blueprint

from .views import *

bp = Blueprint("app", __name__, template_folder='templates', static_folder='static')

bp.add_url_rule('/', view_func=IndexView.as_view('index'))

bp.add_url_rule('/', view_func=PredictView.as_view('predict'))