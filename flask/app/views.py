from flask import render_template, request, jsonify
from flask.views import View, MethodView

from .utils import predict

class IndexView(View):
    def __init__(self) -> None:
        self.template = 'index.html'

    def dispatch_request(self):
        return render_template(self.template)
    
class PredictView(MethodView):
    init_every_request = False

    def _validate_request(self, request):

        text = None

        text = request.json.get('text')
        if text is not None: return self._validate_text(text)

        return False
    
    def _validate_text(self, text: str):
        if text.strip() == "":
            return False
        return True
    
    def _get_text(self, request):
        text = request.json.get('text')
        if text is not None: return text

    def post(self):
        text = None
        validated_request = self._validate_request(request)
        if not validated_request:
            return jsonify({'success': False, 'message': "Invalid input"}), 400
        
        text = self._get_text(request)

        try:
            sentiment, probability = predict(text)
        except:
            print('Error in predicting')
            return jsonify({'success': False}), 500
        
        print(f"Sentiment: {sentiment}, Probability: {probability}")

        return jsonify({'success': True,
                        'sentiment': sentiment,
                        'probability': probability}), 200