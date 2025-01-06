'use client';

import { useState } from 'react';

const DrugAnalysis = () => {
  const [formData, setFormData] = useState({
    Molecular_Weight: '',
    H_Bond_Donors: '',
    H_Bond_Acceptors: '',
    Polar_Surface_Area: '',
    Number_of_Rings: '',
    Rotatable_Bonds: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          ...formData,
          Molecular_Weight: parseFloat(formData.Molecular_Weight),
          H_Bond_Donors: parseInt(formData.H_Bond_Donors),
          H_Bond_Acceptors: parseInt(formData.H_Bond_Acceptors),
          Polar_Surface_Area: parseFloat(formData.Polar_Surface_Area),
          Number_of_Rings: parseInt(formData.Number_of_Rings),
          Rotatable_Bonds: parseInt(formData.Rotatable_Bonds)
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get prediction');
      }

      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-2xl font-bold mb-6 text-gray-800">Enter Compound Details</h2>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.keys(formData).map((key) => (
            <div key={key}>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                {key.replace(/_/g, ' ')}
              </label>
              <input
                type="number"
                step="any"
                className="w-full p-2 border rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900"
                value={formData[key]}
                onChange={(e) => setFormData(prev => ({
                  ...prev,
                  [key]: e.target.value
                }))}
                required
              />
            </div>
          ))}
        </div>
        
        <button
          type="submit"
          disabled={loading}
          className="w-full bg-blue-600 text-white p-3 rounded-md hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
        >
          {loading ? 'Analyzing...' : 'Analyze Compound'}
        </button>
      </form>

      {error && (
        <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-md">
          {error}
        </div>
      )}

      {prediction && (
        <div className="mt-6 p-6 bg-gray-50 rounded-md">
          <h3 className="text-xl font-semibold mb-4 text-gray-800">Analysis Results</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center p-3 bg-white rounded shadow">
              <span className="font-medium">Prediction:</span>
              <span className={`px-3 py-1 rounded ${
                prediction.prediction === 'High_Potency' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-yellow-100 text-yellow-800'
              }`}>
                {prediction.prediction}
              </span>
            </div>
            <div className="flex justify-between items-center p-3 bg-white rounded shadow">
              <span className="font-medium">Confidence:</span>
              <span className="text-blue-600 font-bold">
                {prediction.probability}%
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DrugAnalysis;