import pickle
from flask import Flask, jsonify, request
from penguin_predict_service import predict_single

app = Flask('penguin-predict-knn')

with open('../models/penguin_knn.pkl', 'rb') as f:
    dv, scaler, model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    penguin_data = request.get_json()
    prediction, probabilities = predict_single(penguin_data, dv, scaler, model)
    
    result = {
        'model': 'KNN',
        'predicted_species': prediction,
        'probabilities': {
            'Adelie': float(probabilities[0]) if probabilities is not None else None,
            'Chinstrap': float(probabilities[1]) if probabilities is not None else None,
            'Gentoo': float(probabilities[2]) if probabilities is not None else None
        } if probabilities is not None else None
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=8004)