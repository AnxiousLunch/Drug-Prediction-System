'use client';

import { useState, useRef, useEffect } from 'react';

const MoleculeBuilder = () => {
  const canvasRef = useRef(null);
  const [atoms, setAtoms] = useState([]);
  const [bonds, setBonds] = useState([]);
  const [selectedAtom, setSelectedAtom] = useState(null);
  const [selectedTool, setSelectedTool] = useState('carbon');
  const [bondType, setBondType] = useState('single');
  const [prediction, setPrediction] = useState(null);
  const [tempLine, setTempLine] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const exampleMolecules = [
    {
      id: 'high_potency_1',
      name: 'High Potency Example 1',
      description: 'Compound 2 - Complex structure with optimal features',
      features: {
        Molecular_Weight: 465.99,
        H_Bond_Donors: 9,
        H_Bond_Acceptors: 2,
        Polar_Surface_Area: 225.48,
        Number_of_Rings: 4,
        Rotatable_Bonds: 5
      },
    },
    {
      id: 'high_potency_2',
      name: 'High Potency Example 2',
      description: 'Compound 19 - Balanced molecule profile',
      features: {
        Molecular_Weight: 245.61,
        H_Bond_Donors: 7,
        H_Bond_Acceptors: 7,
        Polar_Surface_Area: 74.13,
        Number_of_Rings: 2,
        Rotatable_Bonds: 9
      },
    },
    {
      id: 'high_potency_3',
      name: 'High Potency Example 3',
      description: 'Compound 33 - High molecular weight',
      features: {
        Molecular_Weight: 574.44,
        H_Bond_Donors: 2,
        H_Bond_Acceptors: 4,
        Polar_Surface_Area: 182.24,
        Number_of_Rings: 1,
        Rotatable_Bonds: 12
      },
    },
    {
      id: 'low_potency_1',
      name: 'Low Potency Example 1',
      description: 'Compound 6 - Simple structure',
      features: {
        Molecular_Weight: 129.04,
        H_Bond_Donors: 4,
        H_Bond_Acceptors: 8,
        Polar_Surface_Area: 98.29,
        Number_of_Rings: 3,
        Rotatable_Bonds: 0
      },
    },
    {
      id: 'low_potency_2',
      name: 'Low Potency Example 2',
      description: 'Compound 61 - Limited bonds',
      features: {
        Molecular_Weight: 235.67,
        H_Bond_Donors: 3,
        H_Bond_Acceptors: 5,
        Polar_Surface_Area: 136.60,
        Number_of_Rings: 0,
        Rotatable_Bonds: 6
      },
    },
    {
      id: 'low_potency_3',
      name: 'Low Potency Example 3',
      description: 'Compound 44 - Simple ring structure',
      features: {
        Molecular_Weight: 229.38,
        H_Bond_Donors: 0,
        H_Bond_Acceptors: 7,
        Polar_Surface_Area: 140.51,
        Number_of_Rings: 1,
        Rotatable_Bonds: 4
      },
    }
  ];

  const availableAtoms = [
    { symbol: 'C', name: 'Carbon', color: '#909090' },
    { symbol: 'O', name: 'Oxygen', color: '#FF0000' },
    { symbol: 'N', name: 'Nitrogen', color: '#3050F8' },
    { symbol: 'H', name: 'Hydrogen', color: '#FFFFFF' },
    { symbol: 'S', name: 'Sulfur', color: '#FFC832' },
  ];

  const bondTypes = [
    { type: 'single', name: 'Single Bond', symbol: '—' },
    { type: 'double', name: 'Double Bond', symbol: '=' },
    { type: 'triple', name: 'Triple Bond', symbol: '≡' },
    { type: 'aromatic', name: 'Aromatic Bond', symbol: '⚫' },
  ];


  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Check if clicking on an existing atom
    const clickedAtom = atoms.find((a) => 
      Math.hypot(a.x - x, a.y - y) <= 25
    );

    if (selectedTool === 'bond') {
      if (clickedAtom) {
        if (!selectedAtom) {
          // First atom selection
          setSelectedAtom(clickedAtom.id);
        } else {
          // Second atom selection - create bond
          if (clickedAtom.id !== selectedAtom) {
            const bondExists = bonds.some(
              bond => (bond.atom1 === selectedAtom && bond.atom2 === clickedAtom.id) ||
                     (bond.atom1 === clickedAtom.id && bond.atom2 === selectedAtom)
            );
            
            if (!bondExists) {
              setBonds(prevBonds => [
                ...prevBonds,
                { 
                  atom1: selectedAtom, 
                  atom2: clickedAtom.id, 
                  type: bondType 
                }
              ]);
            }
            setSelectedAtom(null);
            setTempLine(null);
          }
        }
      }
    } else {
      // If not a bond tool and not clicking on an existing atom
      if (!clickedAtom) {
        const newAtom = {
          id: `atom_${atoms.length}`,
          symbol: selectedTool.toUpperCase(),
          color: availableAtoms.find((a) => a.symbol.toLowerCase() === selectedTool).color,
          x,
          y,
        };
        setAtoms(prevAtoms => [...prevAtoms, newAtom]);
      }
    }
  };

  const handleMouseMove = (e) => {
    if (!selectedAtom) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    setTempLine({ 
      x1: atoms.find((a) => a.id === selectedAtom).x, 
      y1: atoms.find((a) => a.id === selectedAtom).y, 
      x2: x, 
      y2: y 
    });
  };


  const clearCanvas = () => {
    setAtoms([]);
    setBonds([]);
    setPrediction(null);
    setSelectedAtom(null);
    setTempLine(null);
  };


  const loadExample = (example) => {
    // Clear existing atoms and bonds
    setAtoms([]);
    setBonds([]);
  
    // Define unique atom and bond configurations for each example
    const exampleConfigurations = {
      high_potency_1: {
        atoms: [
          { id: 'atom_0', symbol: 'C', color: availableAtoms.find(a => a.symbol === 'C').color, x: 300, y: 200 },
          { id: 'atom_1', symbol: 'O', color: availableAtoms.find(a => a.symbol === 'O').color, x: 350, y: 200 },
          { id: 'atom_2', symbol: 'N', color: availableAtoms.find(a => a.symbol === 'N').color, x: 250, y: 200 },
        ],
        bonds: [
          { atom1: 'atom_0', atom2: 'atom_1', type: 'double' },
          { atom1: 'atom_0', atom2: 'atom_2', type: 'single' },
        ],
      },
      high_potency_2: {
        atoms: [
          { id: 'atom_0', symbol: 'C', color: availableAtoms.find(a => a.symbol === 'C').color, x: 300, y: 200 },
          { id: 'atom_1', symbol: 'C', color: availableAtoms.find(a => a.symbol === 'C').color, x: 350, y: 250 },
          { id: 'atom_2', symbol: 'O', color: availableAtoms.find(a => a.symbol === 'O').color, x: 250, y: 150 },
        ],
        bonds: [
          { atom1: 'atom_0', atom2: 'atom_1', type: 'single' },
          { atom1: 'atom_0', atom2: 'atom_2', type: 'double' },
        ],
      },
      high_potency_3: {
        atoms: [
          { id: 'atom_0', symbol: 'C', color: availableAtoms.find(a => a.symbol === 'C').color, x: 300, y: 200 },
          { id: 'atom_1', symbol: 'O', color: availableAtoms.find(a => a.symbol === 'O').color, x: 350, y: 200 },
          { id: 'atom_2', symbol: 'N', color: availableAtoms.find(a => a.symbol === 'N').color, x: 250, y: 200 },
          { id: 'atom_3', symbol: 'H', color: availableAtoms.find(a => a.symbol === 'H').color, x: 300, y: 250 },
        ],
        bonds: [
          { atom1: 'atom_0', atom2: 'atom_1', type: 'double' },
          { atom1: 'atom_0', atom2: 'atom_2', type: 'single' },
          { atom1: 'atom_0', atom2: 'atom_3', type: 'single' },
        ],
      },
      low_potency_1: {
        atoms: [
          { id: 'atom_0', symbol: 'C', color: availableAtoms.find(a => a.symbol === 'C').color, x: 300, y: 200 },
          { id: 'atom_1', symbol: 'H', color: availableAtoms.find(a => a.symbol === 'H').color, x: 350, y: 200 },
          { id: 'atom_2', symbol: 'H', color: availableAtoms.find(a => a.symbol === 'H').color, x: 250, y: 200 },
        ],
        bonds: [
          { atom1: 'atom_0', atom2: 'atom_1', type: 'single' },
          { atom1: 'atom_0', atom2: 'atom_2', type: 'single' },
        ],
      },
      low_potency_2: {
        atoms: [
          { id: 'atom_0', symbol: 'C', color: availableAtoms.find(a => a.symbol === 'C').color, x: 300, y: 200 },
          { id: 'atom_1', symbol: 'O', color: availableAtoms.find(a => a.symbol === 'O').color, x: 350, y: 200 },
        ],
        bonds: [
          { atom1: 'atom_0', atom2: 'atom_1', type: 'single' },
        ],
      },
      low_potency_3: {
        atoms: [
          { id: 'atom_0', symbol: 'C', color: availableAtoms.find(a => a.symbol === 'C').color, x: 300, y: 200 },
          { id: 'atom_1', symbol: 'H', color: availableAtoms.find(a => a.symbol === 'H').color, x: 350, y: 250 },
          { id: 'atom_2', symbol: 'H', color: availableAtoms.find(a => a.symbol === 'H').color, x: 250, y: 150 },
          { id: 'atom_3', symbol: 'O', color: availableAtoms.find(a => a.symbol === 'O').color, x: 300, y: 100 },
        ],
        bonds: [
          { atom1: 'atom_0', atom2: 'atom_1', type: 'single' },
          { atom1: 'atom_0', atom2: 'atom_2', type: 'single' },
          { atom1: 'atom_0', atom2: 'atom_3', type: 'double' },
        ],
      },
    };
  
    // Get the configuration for the selected example
    const { atoms: newAtoms, bonds: newBonds } = exampleConfigurations[example.id] || { atoms: [], bonds: [] };
  
    // Set the generated atoms and bonds
    setAtoms(newAtoms);
    setBonds(newBonds);
  
    // Set a prediction based on the example
    setPrediction({
      prediction: example.name,
      probability: 75.0,
      input_features: example.features,
    });
  };
  // Add useEffect to handle canvas rendering
 useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Clear canvas with a slight gradient background
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height);
    gradient.addColorStop(0, '#f8fafc');
    gradient.addColorStop(1, '#f1f5f9');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add grid pattern
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 0.5;
    const gridSize = 40;
    for (let x = 0; x <= canvas.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.height);
      ctx.stroke();
    }
    for (let y = 0; y <= canvas.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.width, y);
      ctx.stroke();
    }
    
    // Draw bonds with shadow
    bonds.forEach(bond => {
      const atom1 = atoms.find(a => a.id === bond.atom1);
      const atom2 = atoms.find(a => a.id === bond.atom2);
      if (atom1 && atom2) {
        ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
        ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.moveTo(atom1.x, atom1.y);
        ctx.lineTo(atom2.x, atom2.y);
        ctx.strokeStyle = '#2c3e50';
        ctx.lineWidth = 3;
        
        if (bond.type === 'double') {
          ctx.setLineDash([4, 4]);
        } else if (bond.type === 'triple') {
          ctx.setLineDash([2, 2]);
        } else {
          ctx.setLineDash([]);
        }
        
        ctx.stroke();
        ctx.shadowBlur = 0;
      }
    });
    
    // Draw atoms with gradient and shadow
    atoms.forEach(atom => {
      ctx.shadowColor = 'rgba(0, 0, 0, 0.2)';
      ctx.shadowBlur = 4;
      ctx.beginPath();
      ctx.arc(atom.x, atom.y, 25, 0, 2 * Math.PI);
      
      const atomGradient = ctx.createRadialGradient(
        atom.x - 5, atom.y - 5, 0,
        atom.x, atom.y, 25
      );
      atomGradient.addColorStop(0, '#ffffff');
      atomGradient.addColorStop(1, atom.color);
      
      ctx.fillStyle = atomGradient;
      ctx.fill();
      ctx.strokeStyle = '#2c3e50';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw atom symbol
      ctx.shadowBlur = 0;
      ctx.fillStyle = atom.color === '#FFFFFF' ? '#2c3e50' : '#ffffff';
      ctx.font = 'bold 20px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(atom.symbol, atom.x, atom.y);
    });
    
    // Draw temporary line if exists
    if (tempLine) {
      ctx.beginPath();
      ctx.moveTo(tempLine.x1, tempLine.y1);
      ctx.lineTo(tempLine.x2, tempLine.y2);
      ctx.strokeStyle = '#94a3b8';
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [atoms, bonds, tempLine]);
  
  const analyzeMolecule = () => {
    setIsAnalyzing(true);
    
    const inputFeatures = {
      Molecular_Weight: atoms.length * 12.01,
      H_Bond_Donors: atoms.filter(atom => atom.symbol === 'O' || atom.symbol === 'N').length,
      H_Bond_Acceptors: atoms.filter(atom => atom.symbol === 'O' || atom.symbol === 'N').length,
      Polar_Surface_Area: atoms.filter(atom => atom.symbol === 'O' || atom.symbol === 'N').length * 20,
      Number_of_Rings: bonds.filter(bond => bond.type === 'aromatic').length,
      Rotatable_Bonds: bonds.length
    };

    setTimeout(() => {
      setPrediction({
        prediction: 'Moderate Potency',
        probability: 68.4,
        input_features: inputFeatures
      });
      setIsAnalyzing(false);
    }, 2000);
  };

  return (
    <div className="flex flex-col items-center bg-gray-50 p-8 rounded-lg shadow-lg max-w-7xl mx-auto">
      <h2 className="text-3xl font-bold text-gray-800 mb-6">Molecule Builder</h2>
      
      <div className="mb-6 flex flex-wrap gap-3 justify-center">
        {availableAtoms.map(atom => (
          <button
            key={atom.symbol}
            onClick={() => setSelectedTool(atom.symbol.toLowerCase())}
            className={`px-4 py-2 rounded-lg font-semibold transition-all ${
              selectedTool === atom.symbol.toLowerCase()
                ? 'bg-blue-500 text-white shadow-lg scale-105'
                : 'bg-white text-gray-800 shadow-md hover:shadow-lg hover:scale-105'
            }`}
          >
            {atom.name}
          </button>
        ))}
        <button
          onClick={() => setSelectedTool('bond')}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            selectedTool === 'bond'
              ? 'bg-blue-500 text-white shadow-lg scale-105'
              : 'bg-white text-gray-800 shadow-md hover:shadow-lg hover:scale-105'
          }`}
        >
          Bond
        </button>
      </div>
      
      {selectedTool === 'bond' && (
        <div className="mb-6 flex gap-3">
          {bondTypes.map(bond => (
            <button
              key={bond.type}
              onClick={() => setBondType(bond.type)}
              className={`px-4 py-2 rounded-lg font-semibold transition-all ${
                bondType === bond.type
                  ? 'bg-green-500 text-white shadow-lg scale-105'
                  : 'bg-white text-gray-800 shadow-md hover:shadow-lg hover:scale-105'
              }`}
            >
              {bond.symbol}
            </button>
          ))}
        </div>
      )}

      <canvas
        ref={canvasRef}
        width={800}
        height={600}
        onClick={handleCanvasClick}
        onMouseMove={handleMouseMove}
        className="border border-gray-200 rounded-lg shadow-xl bg-white mb-6"
      />

      <div className="flex gap-4 mb-6">
        <button
          onClick={analyzeMolecule}
          disabled={isAnalyzing}
          className={`px-6 py-3 rounded-lg font-semibold transition-all ${
            isAnalyzing
              ? 'bg-gray-400 text-white'
              : 'bg-blue-500 text-white shadow-md hover:shadow-lg hover:scale-105'
          }`}
        >
          {isAnalyzing ? 'Analyzing...' : 'Analyze Molecule'}
        </button>
        <button
          onClick={clearCanvas}
          className="px-6 py-3 rounded-lg font-semibold bg-red-500 text-white shadow-md hover:shadow-lg hover:scale-105 transition-all"
        >
          Clear Canvas
        </button>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
        {exampleMolecules.map(example => (
          <button
            key={example.id}
            onClick={() => loadExample(example)}
            className="p-4 bg-white rounded-lg shadow-md hover:shadow-lg transition-all text-gray-800 text-left"
          >
            <h3 className="font-semibold mb-1">{example.name}</h3>
            <p className="text-sm text-gray-600">{example.description}</p>
          </button>
        ))}
      </div>

      {prediction && !prediction.error && (
        <div className="w-full max-w-3xl bg-white rounded-lg shadow-lg p-6">
          <h4 className="text-2xl font-bold text-gray-800 mb-4">Analysis Results</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-gray-50 p-4 rounded-lg">
              <h5 className="text-xl font-semibold text-blue-600 mb-2">Prediction</h5>
              <p className="text-2xl font-bold text-gray-900">{prediction.prediction}</p>
              <p className="text-lg text-gray-600">
                Confidence: {prediction.probability.toFixed(2)}%
              </p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h5 className="text-xl font-semibold text-blue-600 mb-2">Molecular Features</h5>
              <div className="space-y-2">
                {prediction.input_features &&
                  Object.entries(prediction.input_features).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center">
                      <span className="text-gray-600">{key.replace(/_/g, ' ')}:</span>
                      <span className="font-semibold text-gray-900">
                        {typeof value === 'number' ? value.toFixed(2) : value}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default MoleculeBuilder;