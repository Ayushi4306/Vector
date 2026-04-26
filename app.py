# app.py - Complete Flask web application with trained Linear Regression model
# Run with: python app.py
# Then open http://localhost:5000

import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
import os
import random

app = Flask(__name__)

# ============================================================
# GRADIENT DESCENT LINEAR REGRESSION MODEL (from scratch)
# ============================================================

class LinearRegressionGD:
    """
    Linear Regression model trained using Gradient Descent.
    Implements the core math: y = w*x + b
    Loss: Mean Squared Error (MSE)
    Update: w = w - α * dL/dw, b = b - α * dL/db
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=500, verbose=True):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None          # slope (w)
        self.bias = None             # intercept (b)
        self.loss_history = []       # track MSE at each epoch
        self.X_mean = None
        self.X_std = None
        self.is_fitted = False
    
    def _normalize(self, X, fit=True):
        """
        Z-score normalization: (X - mean) / std
        Helps gradient descent converge faster
        """
        if fit:
            self.X_mean = np.mean(X)
            self.X_std = np.std(X)
            if self.X_std == 0:
                self.X_std = 1
        
        return (X - self.X_mean) / self.X_std
    
    def _denormalize_prediction(self, y_norm):
        """
        Convert normalized predictions back to original scale
        For linear regression, prediction doesn't need denormalization of target
        but we keep this method for completeness
        """
        return y_norm
    
    def fit(self, X, y):
        """
        Train the model using Gradient Descent algorithm
        
        Mathematical derivation:
        For y_pred = w*x + b
        MSE = (1/n) * Σ(y_pred - y)²
        
        ∂MSE/∂w = (2/n) * Σ(x * (y_pred - y))
        ∂MSE/∂b = (2/n) * Σ(y_pred - y)
        
        Update rules:
        w = w - α * ∂MSE/∂w
        b = b - α * ∂MSE/∂b
        """
        # Convert to numpy arrays
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        y = np.array(y)
        n_samples = len(X)
        
        # Normalize features for better convergence
        X_norm = self._normalize(X, fit=True)
        
        # Initialize parameters
        self.weights = 0.0
        self.bias = 0.0
        self.loss_history = []
        
        # Gradient Descent loop
        for iteration in range(self.n_iterations):
            # Forward pass: make predictions
            y_pred = self.weights * X_norm.flatten() + self.bias
            
            # Calculate loss (Mean Squared Error)
            mse = np.mean((y_pred - y) ** 2)
            self.loss_history.append(mse)
            
            # Calculate gradients
            dw = (2 / n_samples) * np.sum(X_norm.flatten() * (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Update parameters (move opposite to gradient)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if self.verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: MSE = {mse:.4f}, w = {self.weights:.4f}, b = {self.bias:.4f}")
        
        self.is_fitted = True
        print(f"\n✅ Training completed! Final MSE: {self.loss_history[-1]:.4f}")
        return self
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = np.array(X).reshape(-1, 1) if len(np.array(X).shape) == 1 else np.array(X)
        X_norm = (X - self.X_mean) / self.X_std
        predictions = self.weights * X_norm.flatten() + self.bias
        return predictions
    
    def score(self, X, y):
        """
        Calculate R² score (coefficient of determination)
        R² = 1 - (SS_res / SS_tot)
        Higher is better (max 1.0)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def get_model_params(self):
        """Return model parameters for frontend"""
        return {
            'weight': float(self.weights),
            'bias': float(self.bias),
            'x_mean': float(self.X_mean),
            'x_std': float(self.X_std),
            'loss_history': self.loss_history,
            'is_fitted': self.is_fitted
        }


# ============================================================
# SYNTHETIC HOUSING DATA GENERATION
# ============================================================

def generate_housing_data(n_samples=100, noise=25, random_seed=42):
    """
    Generate realistic synthetic house price data
    
    Formula: Price = 0.13 * Area + 45 + noise
    Area range: 500 - 3500 sq ft
    Price range: ~$110k - $500k
    """
    np.random.seed(random_seed)
    
    # Area in square feet
    area = np.random.uniform(500, 3500, n_samples)
    
    # Base price: $130 per sq ft plus $45k base
    base_price = 0.13 * area + 45
    
    # Add random noise
    noise_values = np.random.normal(0, noise, n_samples)
    price = base_price + noise_values
    
    # Ensure no negative prices
    price = np.maximum(price, 30)
    
    return area, price


# ============================================================
# TRAIN GLOBAL MODEL (runs once when server starts)
# ============================================================

print("🏠 Training House Price Prediction Model...")
print("=" * 50)

# Generate training data
X_train_full, y_train_full = generate_housing_data(n_samples=120, noise=22, random_seed=42)

# Split into train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegressionGD(learning_rate=0.02, n_iterations=500, verbose=True)
model.fit(X_train, y_train)

# Evaluate on validation set
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)

print("\n" + "=" * 50)
print(f"📊 Model Performance on Validation Set:")
print(f"   • Mean Squared Error (MSE): {val_mse:.2f}")
print(f"   • R² Score (accuracy): {val_r2:.4f}")
print(f"   • RMSE: {np.sqrt(val_mse):.2f}")
print("=" * 50)

# Store model parameters for frontend
MODEL_PARAMS = model.get_model_params()
MODEL_PARAMS['final_mse'] = val_mse
MODEL_PARAMS['final_r2'] = val_r2


# ============================================================
# FLASK WEB SERVER
# ============================================================

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    API endpoint for price prediction
    Expects JSON: {"sqft": 1850}
    Returns: {"price": 245.67, "success": true}
    """
    try:
        data = request.get_json()
        sqft = float(data.get('sqft', 0))
        
        if sqft <= 0:
            return jsonify({'error': 'Square footage must be positive', 'success': False}), 400
        
        # Make prediction using trained model
        price_k = model.predict([sqft])[0]
        price = round(price_k, 2)
        
        return jsonify({
            'success': True,
            'sqft': sqft,
            'price': price,
            'price_formatted': f"${price:,.2f}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Return model performance metrics and parameters"""
    return jsonify({
        'success': True,
        'weight': MODEL_PARAMS['weight'],
        'bias': MODEL_PARAMS['bias'],
        'mse': MODEL_PARAMS['final_mse'],
        'r2': MODEL_PARAMS['final_r2'],
        'rmse': np.sqrt(MODEL_PARAMS['final_mse']),
        'is_fitted': MODEL_PARAMS['is_fitted']
    })


@app.route('/api/loss-history', methods=['GET'])
def loss_history():
    """Return training loss history for visualization"""
    return jsonify({
        'success': True,
        'loss_history': MODEL_PARAMS['loss_history'],
        'iterations': len(MODEL_PARAMS['loss_history'])
    })


@app.route('/api/retrain', methods=['POST'])
def retrain():
    """
    Retrain the model with new hyperparameters
    Expects JSON: {"learning_rate": 0.01, "iterations": 300}
    """
    try:
        data = request.get_json()
        lr = float(data.get('learning_rate', 0.02))
        iterations = int(data.get('iterations', 500))
        
        # Generate fresh data
        X_new, y_new = generate_housing_data(n_samples=120, noise=22, random_seed=None)
        X_tr, X_va, y_tr, y_va = train_test_split(X_new, y_new, test_size=0.2, random_state=None)
        
        # Train new model
        global model, MODEL_PARAMS
        new_model = LinearRegressionGD(learning_rate=lr, n_iterations=iterations, verbose=False)
        new_model.fit(X_tr, y_tr)
        
        # Evaluate
        val_pred = new_model.predict(X_va)
        new_mse = mean_squared_error(y_va, val_pred)
        new_r2 = r2_score(y_va, val_pred)
        
        # Update global model
        model = new_model
        MODEL_PARAMS = model.get_model_params()
        MODEL_PARAMS['final_mse'] = new_mse
        MODEL_PARAMS['final_r2'] = new_r2
        
        return jsonify({
            'success': True,
            'message': f'Model retrained! R²: {new_r2:.4f}, MSE: {new_mse:.2f}',
            'mse': new_mse,
            'r2': new_r2,
            'weight': model.weights,
            'bias': model.bias,
            'loss_history': model.loss_history
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict prices for multiple square footage values
    Expects JSON: {"sqft_list": [1000, 1500, 2000]}
    """
    try:
        data = request.get_json()
        sqft_list = data.get('sqft_list', [])
        
        if not sqft_list:
            return jsonify({'error': 'Empty list provided', 'success': False}), 400
        
        predictions = []
        for sqft in sqft_list:
            price = model.predict([float(sqft)])[0]
            predictions.append({
                'sqft': float(sqft),
                'price': round(price, 2)
            })
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Write the HTML template
    html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Predictor • AI Gradient Descent Model</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            background: radial-gradient(circle at 10% 20%, #0A0C12, #030507);
            font-family: 'Inter', sans-serif;
            padding: 2rem;
            color: #EFF3F8;
            min-height: 100vh;
        }
        .container {
            max-width: 1300px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .header h1 {
            font-size: 2.2rem;
            background: linear-gradient(135deg, #FFFFFF, #A0C8F5);
            background-clip: text;
            -webkit-background-clip: text;
            color: transparent;
        }
        .header p {
            color: #9BA4B5;
            margin-top: 0.5rem;
        }
        .grid-3 {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1.2rem;
            margin-bottom: 1.8rem;
        }
        .card {
            background: rgba(18, 24, 34, 0.7);
            backdrop-filter: blur(8px);
            border-radius: 1.5rem;
            border: 1px solid rgba(55, 138, 221, 0.25);
            padding: 1.2rem;
            text-align: center;
        }
        .card .label {
            font-size: 0.7rem;
            text-transform: uppercase;
            color: #7F8C9A;
            letter-spacing: 1px;
        }
        .card .value {
            font-size: 2rem;
            font-weight: 700;
            font-family: 'Fira Code', monospace;
            color: #5FADF0;
        }
        .prediction-box {
            background: linear-gradient(135deg, rgba(55,138,221,0.15), rgba(31,90,158,0.1));
            border-radius: 1.8rem;
            border: 1px solid #378ADD;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
        }
        .input-row {
            display: flex;
            gap: 1rem;
            align-items: flex-end;
            flex-wrap: wrap;
        }
        .input-group {
            flex: 2;
        }
        .input-group label {
            display: block;
            font-size: 0.7rem;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
            color: #9BA4B5;
        }
        .input-group input {
            width: 100%;
            padding: 0.8rem 1rem;
            background: #0A0F17;
            border: 1px solid #2E3A48;
            border-radius: 1rem;
            color: white;
            font-size: 1.1rem;
            font-family: 'Fira Code', monospace;
        }
        .input-group input:focus {
            outline: none;
            border-color: #378ADD;
        }
        .price-result {
            background: rgba(0,0,0,0.4);
            border-radius: 1rem;
            padding: 0.8rem 1.5rem;
            text-align: center;
            min-width: 200px;
        }
        .price-result .prediction-price {
            font-size: 2rem;
            font-weight: 800;
            color: #7AEBC0;
            font-family: 'Fira Code', monospace;
        }
        .btn {
            padding: 0.75rem 1.5rem;
            border-radius: 2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            border: none;
            font-family: 'Inter', sans-serif;
        }
        .btn-primary {
            background: linear-gradient(95deg, #378ADD, #1F5A9E);
            color: white;
        }
        .btn-secondary {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid #378ADD;
            color: #5FADF0;
        }
        .btn:hover {
            transform: translateY(-2px);
            filter: brightness(1.05);
        }
        .chart-container {
            background: rgba(6, 10, 16, 0.7);
            border-radius: 1.5rem;
            border: 1px solid rgba(55,138,221,0.2);
            padding: 1rem;
            margin-bottom: 1.2rem;
        }
        .chart-title {
            font-size: 0.75rem;
            text-transform: uppercase;
            color: #A0B3D1;
            margin-bottom: 0.8rem;
        }
        .training-controls {
            background: rgba(10, 14, 21, 0.7);
            border-radius: 1.2rem;
            padding: 1rem;
            margin: 1rem 0;
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            align-items: center;
        }
        .control-item {
            flex: 1;
        }
        .control-item label {
            font-size: 0.7rem;
            display: flex;
            justify-content: space-between;
        }
        input[type=range] {
            width: 100%;
            height: 4px;
            background: #2D3A4B;
            border-radius: 10px;
        }
        .badge {
            background: #1E293B;
            padding: 0.3rem 0.8rem;
            border-radius: 2rem;
            font-size: 0.7rem;
            font-family: monospace;
        }
        .loading {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        @media (max-width: 768px) {
            .grid-3 { grid-template-columns: 1fr; }
            body { padding: 1rem; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1><i class="fas fa-brain"></i> House Price Predictor</h1>
        <p>Trained with Gradient Descent • Linear Regression from scratch • Real-time predictions</p>
    </div>

    <div class="grid-3">
        <div class="card">
            <div class="label"><i class="fas fa-chart-line"></i> MODEL ACCURACY (R²)</div>
            <div class="value" id="r2-value">—</div>
            <div class="label" style="font-size: 0.65rem;">Higher is better (max 1.0)</div>
        </div>
        <div class="card">
            <div class="label"><i class="fas fa-weight-hanging"></i> WEIGHT (Slope)</div>
            <div class="value" id="weight-value">—</div>
            <div class="label">Price change per sq ft</div>
        </div>
        <div class="card">
            <div class="label"><i class="fas fa-chart-simple"></i> MEAN SQUARED ERROR</div>
            <div class="value" id="mse-value">—</div>
            <div class="label">Lower is better</div>
        </div>
    </div>

    <div class="prediction-box">
        <h3 style="margin-bottom: 1rem;"><i class="fas fa-home"></i> Predict House Price</h3>
        <div class="input-row">
            <div class="input-group">
                <label><i class="fas fa-vector-square"></i> SQUARE FEET (Area)</label>
                <input type="number" id="sqft-input" placeholder="e.g., 1850" value="1850">
            </div>
            <div class="price-result">
                <div style="font-size: 0.7rem; color: #9BA4B5;">PREDICTED PRICE</div>
                <div class="prediction-price" id="predicted-price">$0</div>
            </div>
            <button class="btn btn-primary" id="predict-btn"><i class="fas fa-calculator"></i> Predict</button>
        </div>
    </div>

    <div class="training-controls">
        <div class="control-item">
            <label>⚡ Learning Rate <span id="lr-val">0.02</span></label>
            <input type="range" id="lr-slider" min="1" max="9" value="4" step="1">
        </div>
        <div class="control-item">
            <label>🔄 Iterations <span id="iter-val">500</span></label>
            <input type="range" id="iter-slider" min="100" max="800" value="500" step="50">
        </div>
        <button class="btn btn-secondary" id="retrain-btn"><i class="fas fa-rotate-right"></i> Retrain Model</button>
        <span id="status-badge" class="badge"><i class="fas fa-check-circle"></i> Model Ready</span>
    </div>

    <div class="chart-container">
        <div class="chart-title"><i class="fas fa-chart-line"></i> LOSS CONVERGENCE (MSE during training)</div>
        <div style="height: 220px;"><canvas id="loss-chart"></canvas></div>
    </div>

    <div class="info-tip" style="background: #1A2533; border-radius: 1rem; padding: 0.8rem; font-size: 0.75rem; border-left: 3px solid #378ADD;">
        <i class="fas fa-flask"></i> <strong>How it works:</strong> The model learns the relationship between square footage and house price using <strong>Gradient Descent</strong>. 
        It minimizes Mean Squared Error (MSE) by iteratively adjusting the weight (slope) and bias (intercept). The loss chart shows convergence to the optimal solution.
    </div>
</div>

<script>
    let lossChart = null;
    const lrMap = [0.001, 0.003, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15];

    async function loadModelInfo() {
        try {
            const response = await fetch('/api/model-info');
            const data = await response.json();
            if (data.success) {
                document.getElementById('r2-value').innerHTML = data.r2.toFixed(4);
                document.getElementById('weight-value').innerHTML = data.weight.toFixed(4);
                document.getElementById('mse-value').innerHTML = data.mse.toFixed(2);
            }
        } catch (e) { console.error(e); }
    }

    async function loadLossHistory() {
        try {
            const response = await fetch('/api/loss-history');
            const data = await response.json();
            if (data.success && data.loss_history) {
                drawLossChart(data.loss_history);
            }
        } catch (e) { console.error(e); }
    }

    function drawLossHistoryFromModel(modelLoss) {
        drawLossChart(modelLoss);
    }

    function drawLossChart(lossArray) {
        const ctx = document.getElementById('loss-chart').getContext('2d');
        if (lossChart) lossChart.destroy();
        const labels = lossArray.map((_, i) => i);
        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Training MSE',
                    data: lossArray,
                    borderColor: '#5FADF0',
                    borderWidth: 2,
                    fill: true,
                    backgroundColor: 'rgba(55,138,221,0.1)',
                    pointRadius: 0,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: { legend: { labels: { color: '#9BA4B5' } } },
                scales: {
                    x: { grid: { color: '#1F2A3A' }, ticks: { color: '#9BA4B5' } },
                    y: { grid: { color: '#1F2A3A' }, ticks: { color: '#9BA4B5' } }
                }
            }
        });
    }

    async function predictPrice() {
        const sqft = parseFloat(document.getElementById('sqft-input').value);
        if (isNaN(sqft) || sqft <= 0) {
            document.getElementById('predicted-price').innerHTML = '$?';
            return;
        }
        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sqft: sqft })
            });
            const data = await response.json();
            if (data.success) {
                document.getElementById('predicted-price').innerHTML = `$${data.price.toLocaleString()}`;
            } else {
                document.getElementById('predicted-price').innerHTML = '$Error';
            }
        } catch (e) {
            document.getElementById('predicted-price').innerHTML = '$Error';
        }
    }

    async function retrainModel() {
        const lrIndex = parseInt(document.getElementById('lr-slider').value);
        const learningRate = lrMap[lrIndex];
        const iterations = parseInt(document.getElementById('iter-slider').value);
        
        const statusBadge = document.getElementById('status-badge');
        statusBadge.innerHTML = '<i class="fas fa-sync-alt fa-spin"></i> Training...';
        statusBadge.classList.add('loading');
        
        try {
            const response = await fetch('/api/retrain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ learning_rate: learningRate, iterations: iterations })
            });
            const data = await response.json();
            if (data.success) {
                document.getElementById('r2-value').innerHTML = data.r2.toFixed(4);
                document.getElementById('weight-value').innerHTML = data.weight.toFixed(4);
                document.getElementById('mse-value').innerHTML = data.mse.toFixed(2);
                if (data.loss_history) drawLossHistoryFromModel(data.loss_history);
                statusBadge.innerHTML = '<i class="fas fa-check-circle"></i> Retrained!';
                setTimeout(() => {
                    statusBadge.innerHTML = '<i class="fas fa-check-circle"></i> Model Ready';
                }, 2000);
                predictPrice();
            } else {
                statusBadge.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Error';
            }
        } catch (e) {
            statusBadge.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Failed';
        }
        statusBadge.classList.remove('loading');
    }

    document.getElementById('predict-btn').addEventListener('click', predictPrice);
    document.getElementById('sqft-input').addEventListener('input', predictPrice);
    document.getElementById('retrain-btn').addEventListener('click', retrainModel);
    
    document.getElementById('lr-slider').addEventListener('input', function() {
        const idx = parseInt(this.value);
        document.getElementById('lr-val').innerText = lrMap[idx];
    });
    document.getElementById('iter-slider').addEventListener('input', function() {
        document.getElementById('iter-val').innerText = this.value;
    });

    loadModelInfo();
    loadLossHistory();
    setTimeout(predictPrice, 500);
</script>
</body>
</html>'''
    
    # Write template file
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("\n🚀 Starting Flask server...")
    print("📍 Open http://localhost:5000 in your browser")
    print("📡 API endpoints available:")
    print("   POST /api/predict - Predict house price")
    print("   GET  /api/model-info - Get model metrics")
    print("   GET  /api/loss-history - Get training loss")
    print("   POST /api/retrain - Retrain with new hyperparameters")
    print("   POST /api/batch-predict - Batch predictions")
    print("\n" + "=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)