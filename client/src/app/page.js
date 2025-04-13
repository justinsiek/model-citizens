'use client'

import React, { useState } from 'react';

function Index() {
  const [prompt, setPrompt] = useState('Do you like pinapple on pizza?');
  const [aiResponse, setAiResponse] = useState('yes I love pizza with pinapple');
  const [userResponse, setUserResponse] = useState('');
  const [results, setResults] = useState([]);

  // Array of 50 different person emojis
  const personEmojis = [
    '👨', '👩', '👶', '👴', '👵', '👱', '👮', '👷', '💂', '🕵️',
    '👨‍⚕️', '👩‍⚕️', '👨‍🎓', '👩‍🎓', '👨‍🏫', '👩‍🏫', '👨‍⚖️', '👩‍⚖️', '👨‍🌾', '👩‍🌾',
    '👨‍🍳', '👩‍🍳', '👨‍🔧', '👩‍🔧', '👨‍🏭', '👩‍🏭', '👨‍💼', '👩‍💼', '👨‍🔬', '👩‍🔬',
    '👨‍💻', '👩‍💻', '👨‍🚀', '👩‍🚀', '👨‍🚒', '👩‍🚒', '👳', '🧕', '👲', '🧔',
    '🤵', '👰', '🤰', '🤱', '👼', '🎅', '🤶', '🧙', '🧚', '🧛'
  ];

  const predict = () => {
    fetch(`http://127.0.0.1:5000/api/predict?user_response=${userResponse}&prompt=${prompt}&ai_response=${aiResponse}`)
      .then(response => {
        return response.json();
      })
      .then(data => {
        setResults(data.result);
      });
  };

  return (
    <div className="min-h-screen bg-white text-gray-900 p-6">
      <div className="w-full max-w-6xl mx-auto pt-6 pb-8">
        {/* Header */}
        <div className="mb-10 border-b border-gray-200 pb-4">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <h1 className="text-3xl font-bold tracking-tight mb-4 md:mb-0">Model Citizens</h1>
            <div className="bg-gray-100 px-4 py-2 rounded-md inline-block">
              <h3 className="text-sm font-medium text-gray-700">
                {prompt}
              </h3>
            </div>
          </div>
        </div>
        
        {/* Response Containers */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Person 1 Response Section */}
          <div className="bg-white rounded-lg p-6 shadow-md border border-gray-200">
            <h4 className="text-lg font-semibold mb-3 flex items-center text-gray-900">
              <span className="bg-black text-white p-1 rounded-md mr-2 text-sm w-6 h-6 flex items-center justify-center">1</span>
              Person 1
            </h4>
            <textarea
              className="w-full bg-gray-50 text-gray-800 border border-gray-300 p-4 rounded-md mb-4 h-48 focus:border-gray-500 focus:ring-1 focus:ring-gray-500 focus:outline-none resize-none"
              value={userResponse}
              onChange={(e) => setUserResponse(e.target.value)}
              placeholder="Enter your response..."
            />
            <div className="bg-gray-50 p-3 rounded-md min-h-16 flex flex-wrap gap-2 items-center">
              {results.length > 0 ? (
                results.map((probability, index) => (
                  probability >= 0.5 ? 
                    <div key={index} className="relative group">
                      <span className="text-xl bg-gray-100 p-1 rounded-md inline-block border border-gray-200">
                        {personEmojis[index % personEmojis.length]}
                      </span>
                      <div className="absolute -top-10 left-0 bg-black text-white p-1 rounded-sm opacity-0 group-hover:opacity-100 transition-opacity text-xs whitespace-nowrap">
                        Score: {probability.toFixed(2)}
                      </div>
                    </div> : null
                ))
              ) : (
                <span className="text-gray-500 text-sm">Human-like responses will appear here</span>
              )}
            </div>
          </div>
          
          {/* Person 2 Response Section */}
          <div className="bg-white rounded-lg p-6 shadow-md border border-gray-200">
            <h4 className="text-lg font-semibold mb-3 flex items-center text-gray-900">
              <span className="bg-black text-white p-1 rounded-md mr-2 text-sm w-6 h-6 flex items-center justify-center">2</span>
              Person 2
            </h4>
            <textarea
              className="w-full bg-gray-50 text-gray-800 border border-gray-300 p-4 rounded-md mb-4 h-48 focus:border-gray-500 focus:ring-1 focus:ring-gray-500 focus:outline-none resize-none"
              value={aiResponse}
              onChange={(e) => setAiResponse(e.target.value)}
              placeholder="Enter response..."
            />
            <div className="bg-gray-50 p-3 rounded-md min-h-16 flex flex-wrap gap-2 items-center">
              {results.length > 0 ? (
                results.map((probability, index) => (
                  probability < 0.5 ? 
                    <div key={index} className="relative group">
                      <span className="text-xl bg-gray-100 p-1 rounded-md inline-block border border-gray-200">
                        {personEmojis[index % personEmojis.length]}
                      </span>
                      <div className="absolute -top-10 left-0 bg-black text-white p-1 rounded-sm opacity-0 group-hover:opacity-100 transition-opacity text-xs whitespace-nowrap">
                        Score: {probability.toFixed(2)}
                      </div>
                    </div> : null
                ))
              ) : (
                <span className="text-gray-500 text-sm">AI-like responses will appear here</span>
              )}
            </div>
          </div>
        </div>
        
        {/* Predict Button */}
        <div className="flex justify-center">
          <button
            onClick={predict}
            className="bg-black hover:bg-gray-800 text-white font-medium py-2 px-8 rounded-md text-sm shadow-sm transition-colors">
            Predict
          </button>
        </div>
      </div>
    </div>
  );
}

export default Index;