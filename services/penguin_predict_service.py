def predict_single(penguin_data, dv, scaler, model):
    x_encoded = dv.transform([penguin_data])
    x_scaled = scaler.transform(x_encoded)
    prediction = model.predict(x_scaled)[0]
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(x_scaled)[0]
    else:
        probabilities = None
    
    return prediction, probabilities