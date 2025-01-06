import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def generate_synthetic_drug_dataset(num_samples=1129):
    """
    Generate a synthetic dataset of drug and antibiotic properties
    """
    np.random.seed(42)
    
    # Generate features
    data = {
        'Compound_ID': [f'Compound_{i}' for i in range(num_samples)],
        'Molecular_Weight': np.random.uniform(100, 600, num_samples),
        'H_Bond_Donors': np.random.randint(0, 10, num_samples),
        'H_Bond_Acceptors': np.random.randint(0, 10, num_samples),
        'Polar_Surface_Area': np.random.uniform(0, 250, num_samples),
        'Number_of_Rings': np.random.randint(0, 6, num_samples),
        'Rotatable_Bonds': np.random.randint(0, 15, num_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variables with some correlation
    df['Antibiotic_Effectiveness'] = (
        0.3 * df['Molecular_Weight'] / 600 + 
        0.2 * df['H_Bond_Donors'] / 10 + 
        0.2 * df['Polar_Surface_Area'] / 250 + 
        0.1 * df['Number_of_Rings'] / 6 + 
        np.random.normal(0, 0.1, num_samples)
    ).clip(0, 1)
    
    df['Drug_Efficacy'] = (
        0.4 * df['Molecular_Weight'] / 600 + 
        0.2 * df['H_Bond_Acceptors'] / 10 + 
        0.2 * df['Rotatable_Bonds'] / 15 + 
        np.random.normal(0, 0.1, num_samples)
    ).clip(0, 1)
    
    # Classify into categories
    df['Antibiotic_Category'] = np.where(
        df['Antibiotic_Effectiveness'] > 0.5, 
        'High_Potency', 'Low_Potency'
    )
    
    df['Drug_Category'] = np.where(
        df['Drug_Efficacy'] > 0.5, 
        'Effective', 'Less_Effective'
    )
    
    # Save to CSV
    df.to_csv('drug_antibiotic_dataset.csv', index=False)
    print("Synthetic dataset created and saved to backend/drug_antibiotic_dataset.csv")
    
    return df

def prepare_graph_data(df, feature_columns, target_column):
    # Preprocess features
    features = df[feature_columns].values
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # Encode labels
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[target_column])
    
    # Create graph data list
    data_list = []
    for i in range(len(df)):
        # Create a simple complete graph for each molecule
        num_nodes = 3  # Create a small graph with 3 nodes
        x = torch.tensor(np.tile(features[i], (num_nodes, 1)), dtype=torch.float)
        
        # Create fully connected edge index
        edge_index = []
        for j in range(num_nodes):
            for k in range(j+1, num_nodes):
                edge_index.extend([[j, k], [k, j]])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        y = torch.tensor(labels[i], dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    
    return data_list, label_encoder, scaler

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

def train_gnn_model(train_loader, in_channels, num_classes, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(in_channels, hidden_channels=64, out_channels=num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            # Ensure batch is on the correct device
            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            # Compute outputs and loss
            outputs = model(batch)
            
            # Reshape the outputs and targets to match the expected shape for CrossEntropyLoss
            # Ensure `outputs` has shape (batch_size, num_classes) and `y` has shape (batch_size,)
            loss = loss_fn(outputs.view(-1, num_classes), batch.y.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')
    torch.save(model.state_dict(), 'gnn_model.pth')
    print("Model saved to 'gnn_model.pth'")
    return model

def evaluate_model(model, test_loader, label_encoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            out = model(batch)  # This is a scalar in your case due to global mean pooling
            
            # Directly get the predicted class from the output (since it's a scalar)
            pred = torch.argmax(out, dim=0).item()  # Getting the class index
            
            predictions.append(pred)
            true_labels.append(batch.y.item())
    
    # Convert predictions back to original labels
    predicted_labels = label_encoder.inverse_transform(predictions)
    true_original_labels = label_encoder.inverse_transform(true_labels)
    
    # Calculate accuracy
    accuracy = np.mean(np.array(predictions) == np.array(true_labels))
    print(f"Model Accuracy: {accuracy:.2%}")
    
    # Detailed classification report
    from sklearn.metrics import classification_report
    print("\nClassification Report:")
    print(classification_report(true_original_labels, predicted_labels))


def main():
    # Generate or load synthetic dataset
    df = generate_synthetic_drug_dataset()
    
    # Choose prediction target (Antibiotic_Category or Drug_Category)
    feature_columns = [
        'Molecular_Weight', 
        'H_Bond_Donors', 
        'H_Bond_Acceptors', 
        'Polar_Surface_Area', 
        'Number_of_Rings', 
        'Rotatable_Bonds'
    ]
    target_column = 'Antibiotic_Category'  # or 'Drug_Category'
    
    # Prepare graph data
    data_list, label_encoder, feature_scaler = prepare_graph_data(
        df, feature_columns, target_column
    )
    
    # Split data
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Train the model
    model = train_gnn_model(
        train_loader, 
        in_channels=len(feature_columns), 
        num_classes=len(label_encoder.classes_)
    )
    
    # Evaluate the model
    evaluate_model(model, test_loader, label_encoder)

if __name__ == "__main__":
    main()