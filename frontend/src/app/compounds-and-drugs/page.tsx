'use client';

export default function DrugAndCompounds() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 text-gray-100">
      {/* Header Section */}
      <div className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/50 to-purple-900/50 animate-pulse"></div>
        <div className="relative z-10 max-w-5xl text-center px-4">
          <h1 className="text-6xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent animate-fadeIn">
            Compounds & Bonds
          </h1>
          <p className="text-xl md:text-2xl text-gray-200 mb-10 max-w-3xl mx-auto">
            Explore the science of compounds, bond types, and our cutting-edge Graph Neural Network (GNN) approach for predicting molecular interactions.
          </p>
        </div>
      </div>

      {/* Compounds Overview Section */}
      <section className="py-20 bg-gray-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            What Are Compounds?
          </h2>
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div className="text-gray-300 text-lg leading-relaxed mb-6 transform hover:scale-110 hover:rotate-3d transition-all duration-500">
              <p className="mb-6">
                Compounds are substances formed by the chemical combination of two or more elements. These molecules have specific properties and are essential for various biological, chemical, and industrial processes.
              </p>
              <p>
                Our platform leverages advanced computational techniques to analyze and predict compound behavior, enabling faster drug discovery and material development. By studying molecular structures, researchers can identify and synthesize compounds with targeted therapeutic effects, reducing time and costs in drug development.
              </p>
            </div>
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-3xl shadow-2xl p-8 text-center hover:scale-105 hover:rotate-3d transition-transform duration-500">
              <h3 className="text-3xl font-semibold mb-4 text-white">Compound Exploration</h3>
              <p className="text-gray-100">
                Explore how compounds interact, behave, and contribute to therapeutic effects in various treatments.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Types of Bonds Section */}
      <section className="py-20 bg-gray-800">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Types of Bonds
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[ 
              { title: 'Covalent Bonds', description: 'Formed by sharing electrons between atoms, creating strong and stable molecules.' },
              { title: 'Ionic Bonds', description: 'Created by the electrostatic attraction between positively and negatively charged ions.' },
              { title: 'Hydrogen Bonds', description: 'Weak bonds formed by the attraction between hydrogen and more electronegative atoms like oxygen or nitrogen.' },
              { title: 'Metallic Bonds', description: 'Occur in metals where electrons are shared collectively in a "sea" of electrons.' },
              { title: 'Van der Waals Forces', description: 'Weak intermolecular forces that arise from transient dipoles in molecules.' }
            ].map((bond, index) => (
              <div key={index} className="bg-gray-700/50 backdrop-blur-sm rounded-2xl p-6 text-center shadow-xl transform hover:scale-110 hover:rotate-3d transition-all duration-500">
                <h3 className="text-xl font-semibold text-blue-400 mb-4">{bond.title}</h3>
                <p className="text-gray-300">{bond.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* GNN Approach Section */}
      <section className="py-20 bg-gray-900/50">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Our GNN Approach
          </h2>
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div className="text-gray-300 text-lg leading-relaxed transform hover:scale-110 hover:rotate-3d transition-all duration-500">
              <p className="mb-6">
                We use Graph Neural Networks (GNNs) to model molecular structures as graphs. Nodes represent atoms, while edges denote bonds. This allows us to predict molecular properties, interactions, and potential applications in drug discovery.
              </p>
              <p className="mb-6">
                Our platform visualizes molecular graphs, highlights key interactions, and predicts minimum &aps;friends&aps; or related compounds you might need for your research. Using this approach, we can uncover hidden relationships between compounds that would take much longer to discover manually.
              </p>
              <p>
                The result? A revolutionary step forward in understanding molecular behavior and accelerating innovation in the pharmaceutical industry.
              </p>
            </div>
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 rounded-lg shadow-2xl p-8 text-center hover:scale-105 hover:rotate-3d transition-transform duration-500">
              <h3 className="text-3xl font-semibold mb-4 text-white">GNN Molecular Visualization</h3>
              <p className="text-gray-100">
                Discover how GNN visualizations can help predict molecular behavior and interactions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Backend Overview Section */}
      <section className="py-20 bg-gray-800">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Backend Technology & Drug Predictions
          </h2>
          <div className="text-gray-300 text-lg leading-relaxed mb-6">
            <p className="mb-4">
              Our backend leverages cutting-edge technologies like Flask, PyTorch, and PyTorch Geometric to provide real-time drug prediction capabilities. Using deep learning models like Graph Neural Networks (GNNs), we process molecular data and provide predictions on compounds&#39; effectiveness and their potential interactions with drugs.
            </p>
            <p className="mb-4">
              The data inputs used in our model include critical features like molecular weight, hydrogen bond donors, hydrogen bond acceptors, polar surface area, the number of rings, and the number of rotatable bonds. These features are fed into the model to predict whether a compound is more likely to exhibit low or high drug efficacy, as well as its potential interaction with antibiotics.
            </p>
            <p>
              By utilizing this system, researchers and pharmaceutical companies can more quickly identify promising compounds for further testing, leading to faster drug discovery and improved treatments for various diseases.
            </p>
          </div>
          <div className="flex justify-center mt-8">
            <a href="#predict" className="px-6 py-3 text-lg font-semibold text-white bg-blue-600 rounded-md hover:bg-blue-500 transition duration-200 transform hover:scale-105">Try Our Prediction Tool</a>
          </div>
        </div>
      </section>

      {/* Benefits & Societal Impact Section */}
      <section className="py-20 bg-gray-900">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Societal Benefits & Applications
          </h2>
          <div className="text-gray-300 text-lg leading-relaxed mb-6">
            <p className="mb-4">
              The application of our GNN model to drug discovery has the potential to drastically accelerate the development of new medicines. This will reduce the time it takes to find life-saving drugs and lower costs associated with traditional drug development methods.
            </p>
            <p className="mb-4">
              In the medical field, this technology could help identify novel antibiotics, anticancer compounds, and treatments for chronic diseases. Beyond pharmaceuticals, it also has applications in material science, where compounds with desirable properties can be synthesized for use in various industries, including electronics, energy, and manufacturing.
            </p>
            <p>
              Ultimately, this project offers a leap forward in not only drug development but also in the broader field of computational chemistry, making science more accessible and efficient, ultimately benefiting global health and well-being.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
