from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import joblib

# Define the GNN model class
class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        
        # Global pooling (mean across nodes)
        x = x.mean(dim=0)  # Aggregate node features
        x = self.fc(x)
        return x

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model
try:
    model = GNNModel(in_channels=6, hidden_channels=64, out_channels=2)
    model.load_state_dict(torch.load('gnn_model.pth', map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Initialize or load preprocessing objects
try:
    scaler = joblib.load('scaler.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except:
    print("Creating new preprocessing objects")
    scaler = StandardScaler()
    label_encoder = LabelEncoder()
    label_encoder.fit(["Low_Potency", "High_Potency"])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        if request.method == 'GET':
            # For testing: use default values
            data = {
                "Molecular_Weight": 304.22660653534933,
                "H_Bond_Donors": 9,
                "H_Bond_Acceptors": 8,
                "Polar_Surface_Area": 233.60825968836036,
                "Number_of_Rings": 0,
                "Rotatable_Bonds": 11
            }
        else:
            data = request.get_json()

        # Extract features
        features = np.array([[
            data['Molecular_Weight'],
            data['H_Bond_Donors'],
            data['H_Bond_Acceptors'],
            data['Polar_Surface_Area'],
            data['Number_of_Rings'],
            data['Rotatable_Bonds']
        ]])

        # Scale features
        if not hasattr(scaler, 'mean_'):
            scaler.fit(features)
        features_scaled = scaler.transform(features)

        # Create graph data
        num_nodes = 3
        x = torch.tensor(np.tile(features_scaled, (num_nodes, 1)), dtype=torch.float).to(device)
        
        edge_index = []
        for j in range(num_nodes):
            for k in range(j+1, num_nodes):
                edge_index.extend([[j, k], [k, j]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)

        # Create graph data object
        graph_data = Data(x=x, edge_index=edge_index)

        # Get prediction
        with torch.no_grad():
            output = model(graph_data)
            probabilities = torch.softmax(output, dim=0)
            pred_idx = torch.argmax(output, dim=0).item()

        # Get predicted label and probability
        predicted_label = label_encoder.inverse_transform([pred_idx])[0]
        probability = probabilities[pred_idx].item()

        return jsonify({
            'prediction': predicted_label,
            'probability': round(probability * 100, 2),
            'input_features': data
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)