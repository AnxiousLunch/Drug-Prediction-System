'use client';

import React, { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { 
  Beaker, Atom, Brain, Database, Server,  Zap, MapPin, 
  Mail, Phone, Send, 
} from 'lucide-react';

// Dynamic imports for performance
const MoleculeViewer = dynamic(() => import('@/components/MoleculeViewer'), { ssr: false });
const MoleculeBuilder = dynamic(() => import('@/components/MoleculeBuilder'), { ssr: false });

// Animated Counter Component
const AnimatedCounter = ({ end, duration = 2000 }: {end:number, duration?: number}) => {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = 0;
    const timer = setInterval(() => {
      start += Math.ceil(end / 100);
      if (start > end) start = end;
      setCount(start);
      if (start === end) clearInterval(timer);
    }, duration / 100);

    return () => clearInterval(timer);
  }, [end, duration]);

  return <span>{count}</span>;
};

export default function Home() {
  const router = useRouter();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    feedback: '',
    feedbackType: ''
  });

  // Create refs for different sections
  const moleculeViewerRef = useRef<HTMLDivElement>(null);
  const moleculeBuilderRef = useRef<HTMLDivElement>(null);

  // Scroll to a specific section
  const scrollToSection = (elementRef: React.RefObject<HTMLDivElement | null>) => {
    elementRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  // Handle navigation to detailed page
  const navigateToDetailedPage = () => {
    router.push('/compounds-and-drugs');
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormData(prevState => ({...prevState, [name]: value}));
  };

  const handleSubmit = (e: React.ChangeEvent<HTMLFormElement>) => {
    e.preventDefault();
    // Add form submission logic
    console.log('Feedback Submitted:', formData);
    // Reset form after submission
    setFormData({
      name: '',
      email: '',
      feedback: '',
      feedbackType: ''
    });
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-900 via-gray-800 to-gray-900 text-gray-100 overflow-x-hidden">
      {/* Hero Section */}
      <div className="relative min-h-screen flex items-center justify-center">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-900/50 to-purple-900/50 animate-pulse"></div>
        <div className="relative z-10 max-w-4xl text-center px-4">
          <h1 className="text-6xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-blue-400 bg-clip-text text-transparent animate-fadeIn">
            Drug Prediction AI
          </h1>
          <p className="text-xl md:text-2xl text-gray-200 mb-10 max-w-2xl mx-auto">
            Revolutionizing pharmaceutical research through advanced molecular modeling and artificial intelligence
          </p>
          <div className="flex justify-center space-x-6">
            <button 
              onClick={() => scrollToSection(moleculeViewerRef)}
              className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-full transition transform hover:scale-105 shadow-2xl"
            >
              Explore Platform
            </button>
            <button 
              onClick={navigateToDetailedPage}
              className="bg-transparent border-2 border-white/30 text-white px-8 py-3 rounded-full hover:bg-white/10 transition transform hover:scale-105"
            >
              Learn More
            </button>
          </div>
        </div>
      </div>

      {/* Molecule Viewer Section */}
      <section 
        ref={moleculeViewerRef} 
        className="py-20 bg-gray-900/50 backdrop-blur-sm"
      >
        <div className="container mx-auto px-4">
          <div className="relative h-[600px] w-full rounded-2xl bg-gray-800/50 shadow-2xl overflow-hidden border-2 border-gray-700/50">
            <MoleculeViewer />
            <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-gray-900/80 to-transparent" />
          </div>
        </div>
      </section>


      {/* Impact Statistics Section */}
      <section className="py-20 bg-gray-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
            Project Impact
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { 
                icon: <Zap className="w-12 h-12 text-blue-400 mx-auto mb-4" />, 
                value: 81, 
                label: "Prediction Accuracy" 
              },
              { 
                icon: <Database className="w-12 h-12 text-purple-400 mx-auto mb-4" />, 
                value: 500, 
                label: "Molecular Compounds Analyzed" 
              },
              { 
                icon: <Server className="w-12 h-12 text-green-400 mx-auto mb-4" />, 
                value: 3, 
                label: "Faster Drug Discovery" 
              }
            ].map((stat, index) => (
              <div 
                key={index} 
                className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 text-center hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2"
              >
                {stat.icon}
                <h3 className="text-5xl font-bold text-white mb-2">
                  <AnimatedCounter end={stat.value} />%
                </h3>
                <p className="text-gray-300">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Molecule Builder Section */}
      <section 
        ref={moleculeBuilderRef}
        className="py-20 bg-gray-900"
      >
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Interactive Molecule Builder
          </h2>
          <div className="bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 shadow-2xl">
            <MoleculeBuilder />
          </div>
        </div>
      </section>
      
      {/* Feedback Section */}
      <section className="py-20 bg-gray-900">
        <div className="container mx-auto px-4">
          <h2 className="text-4xl font-bold text-center mb-12 bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Share Your Feedback
          </h2>
          <div className="max-w-2xl mx-auto bg-gray-800/50 backdrop-blur-sm rounded-2xl p-8 shadow-2xl border-2 border-gray-700/50">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label htmlFor="name" className="block text-gray-300 mb-2">
                    Name
                  </label>
                  <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 bg-gray-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300"
                    placeholder="Your Name"
                    required
                  />
                </div>
                <div>
                  <label htmlFor="email" className="block text-gray-300 mb-2">
                    Email
                  </label>
                  <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 bg-gray-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-all duration-300"
                    placeholder="your.email@example.com"
                    required
                  />
                </div>
              </div>
              <div>
                <label htmlFor="feedback" className="block text-gray-300 mb-2">
                  Your Feedback
                </label>
                <textarea
                  id="feedback"
                  name="feedback"
                  value={formData.feedback}
                  onChange={handleInputChange}
                  rows={5}
                  className="w-full px-4 py-3 bg-gray-700/50 rounded-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-300"
                  placeholder="Share your thoughts, suggestions, or experiences..."
                  required
                ></textarea>
              </div>
              <div>
                <label className="block text-gray-300 mb-2">
                  Feedback Type
                </label>
                <div className="flex space-x-4">
                  {[
                    { label: "Suggestion", value: "suggestion", color: "text-blue-400" },
                    { label: "Bug Report", value: "bug", color: "text-red-400" },
                    { label: "Praise", value: "praise", color: "text-green-400" }
                  ].map((type, index) => (
                    <label key={index} className="inline-flex items-center">
                      <input
                        type="radio"
                        name="feedbackType"
                        value={type.value}
                        onChange={handleInputChange}
                        className="form-radio h-5 w-5 text-blue-600 bg-gray-800 border-gray-300 focus:ring-blue-500"
                      />
                      <span className={`ml-2 ${type.color}`}>{type.label}</span>
                    </label>
                  ))}
                </div>
              </div>
              <button
                type="submit"
                className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-4 rounded-xl hover:opacity-90 transition-all duration-300 transform hover:scale-105 shadow-xl"
              >
                Submit Feedback
              </button>
            </form>
          </div>
        </div>
      </section>

      {/* Detailed Footer */}
      <footer className="bg-gradient-to-r from-gray-900 to-gray-800 text-gray-300 py-16">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-10">
            {/* Project Overview */}
            <div>
              <h3 className="text-2xl font-bold mb-6 text-white">
                Drug Prediction AI
              </h3>
              <p className="text-gray-400 mb-6">
                Revolutionizing pharmaceutical research through advanced molecular modeling and artificial intelligence.
              </p>
              <div className="flex space-x-4">
                {[
                  { icon: <Atom className="w-6 h-6" />, link: "#" },
                  { icon: <Brain className="w-6 h-6" />, link: "#" },
                  { icon: <Beaker className="w-6 h-6" />, link: "#" }
                ].map((social, index) => (
                  <a
                    key={index}
                    href={social.link}
                    className="text-gray-400 hover:text-white transition-colors"
                  >
                    {social.icon}
                  </a>
                ))}
              </div>
            </div>

            {/* Quick Links */}
            <div>
              <h4 className="text-xl font-bold mb-6 text-white">Quick Links</h4>
              <ul className="space-y-4">
                {[
                  { label: "Team Lead", link: "https://www.linkedin.com/in/syed-omer-ahmed-shamsi-b3aa6a251/" },
                  { label: "ML Engineer", link: "https://www.linkedin.com/in/muhammad-umer-safee-b48256296/" }
                ].map((link, index) => (
                  <li key={index}>
                    <a 
                      href={link.link} 
                      className="hover:text-blue-400 transition-colors"
                    >
                      {link.label}
                    </a>
                  </li>
                ))}
              </ul>
            </div>

            {/* Contact Information */}
            <div>
              <h4 className="text-xl font-bold mb-6 text-white">Contact Us</h4>
              <ul className="space-y-4">
                <li className="flex items-center">
                  <MapPin className="w-5 h-5 mr-3 text-blue-400" />
                  Research Lab, NED University Karachi
                </li>
                <li className="flex items-center">
                  <Mail className="w-5 h-5 mr-3 text-purple-400" />
                  omershamsi911@gmail.com 
                </li>
                <li className="flex items-center">
                  <Mail className="w-5 h-5 mr-3 text-purple-400" />
                  umersafee@gmail.com 
                </li>
                <li className="flex items-center">
                  <Phone className="w-5 h-5 mr-3 text-green-400" />
                  +92 (318) 255-8304
                </li>
                <li className="flex items-center">
                  <Phone className="w-5 h-5 mr-3 text-green-400" />
                  +92 (336) 539-4025
                </li>
              </ul>
            </div>

            {/* Newsletter */}
            <div>
              <h4 className="text-xl font-bold mb-6 text-white">
                Stay Updated
              </h4>
              <p className="text-gray-400 mb-4">
                Subscribe to our newsletter for latest research and updates.
              </p>
              <div className="flex">
                <input
                  type="email"
                  placeholder="Enter your email"
                  className="w-full px-4 py-3 bg-gray-700/50 rounded-l-xl text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button className="bg-blue-600 text-white px-6 rounded-r-xl hover:bg-blue-700 transition-colors">
                  <Send className="w-5 h-5" />
                </button>
              </div>
            </div>
          </div>

          {/* Copyright */}
          <div className="mt-12 pt-8 border-t border-gray-700 text-center">
            <p className="text-gray-500">
              © {new Date().getFullYear()} Drug Prediction AI. All Rights Reserved.
              Developed with ❤️ by Team Arsenic.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}