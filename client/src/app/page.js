'use client'

import React, { useState, useEffect, useRef } from 'react';

function Index() {
  const [prompt, setPrompt] = useState('How do you get better at coding?');
  const [aiResponse, setAiResponse] = useState('');
  const [userResponse, setUserResponse] = useState('');
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [animatedModels, setAnimatedModels] = useState([]);
  const [isAnimating, setIsAnimating] = useState(false);

  // Array of 50 different person emojis
  const personEmojis = [
    '👨', '👩', '👶', '👴', '👵', '👱', '👮', '👷', '💂', '🕵️',
    '👨‍⚕️', '👩‍⚕️', '👨‍🎓', '👩‍🎓', '👨‍🏫', '👩‍🏫', '👨‍⚖️', '👩‍⚖️', '👨‍🌾', '👩‍🌾',
    '👨‍🍳', '👩‍🍳', '👨‍🔧', '👩‍🔧', '👨‍🏭', '👩‍🏭', '👨‍💼', '👩‍💼', '👨‍🔬', '👩‍🔬',
    '👨‍💻', '👩‍💻', '👨‍🚀', '👩‍🚀', '👨‍🚒', '👩‍🚒', '👳', '🧕', '👲', '🧔',
    '🤵', '👰', '🤰', '🤱', '👼', '🎅', '🤶', '🧙', '🧚', '🧛'
  ];

  // Animation handling when results change
  useEffect(() => {
    if (results.length > 0) {
      startModelAnimation();
    }
  }, [results]);

  const startModelAnimation = () => {
    // Reset animation state
    setAnimatedModels([]);
    setIsAnimating(true);
    
    // Create models array from results
    const modelsToAnimate = results.map((probability, index) => ({
      id: index,
      emoji: personEmojis[index % personEmojis.length],
      probability,
      vote: probability >= 0.5 ? 'human' : 'ai'
    }));
    
    // Add models one by one with delay
    modelsToAnimate.forEach((model, index) => {
      setTimeout(() => {
        setAnimatedModels(prev => [...prev, model]);
        
        // Check if this is the last model
        if (index === modelsToAnimate.length - 1) {
          setTimeout(() => setIsAnimating(false), 500);
        }
      }, index * 100); // Adjust delay as needed
    });
  };
  
  const predict = () => {
    setIsLoading(true);
    setAnimatedModels([]);
    fetch(`http://127.0.0.1:5000/api/predict?user_response=${userResponse}&prompt=${prompt}&ai_response=${aiResponse}`)
      .then(response => {
        return response.json();
      })
      .then(data => {
        setResults(data.result);
        setIsLoading(false);
      })
      .catch(error => {
        console.error("Error during prediction:", error);
        setIsLoading(false);
      });
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900 p-0">
      <style jsx global>{`
        .emoji-item {
          opacity: 0;
          transform: scale(0);
          animation: popIn 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }
        
        @keyframes popIn {
          0% {
            opacity: 0;
            transform: scale(0);
          }
          70% {
            opacity: 1;
            transform: scale(1.1);
          }
          100% {
            opacity: 1;
            transform: scale(1);
          }
        }
        
        .emoji-content:hover {
          transform: scale(1.1);
          z-index: 10;
          transition: transform 0.2s ease;
        }
      `}</style>

      <header className="w-full bg-black text-white py-5 px-6 shadow-md">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center">
          <h1 className="text-3xl font-semibold tracking-tight mb-4 md:mb-0">
            model<span className="font-light">citizens</span>
          </h1>
          <div className="bg-gray-800 px-5 py-3 rounded-full inline-block">
            <h3 className="text-sm font-medium text-gray-100">
              "{prompt}"
            </h3>
          </div>
        </div>
      </header>
      
      <div className="w-full max-w-6xl mx-auto pt-6 pb-8">
        {/* Response Containers */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
          {/* Person 1 Response Section */}
          <div className="bg-white rounded-lg p-6 shadow-md border border-black">
            <h4 className="text-lg font-semibold mb-3 flex items-center text-gray-900">
              <span className="bg-black text-white p-1 rounded-md mr-2 text-sm w-6 h-6 flex items-center justify-center">1</span>
              Person 1
            </h4>
            <textarea
              className="w-full bg-gray-50 text-gray-800 border border-gray-300 p-4 rounded-md mb-4 h-48 focus:border-gray-500 focus:ring-1 focus:ring-gray-500 focus:outline-none resize-none"
              value={userResponse}
              onChange={(e) => setUserResponse(e.target.value)}
              placeholder="Enter response..."
            />
            <div className="bg-gray-50 p-3 rounded-md min-h-16 flex flex-wrap gap-2 items-center">
              {animatedModels
                .filter(model => model.vote === 'human')
                .map((model, index) => (
                  <div 
                    key={model.id} 
                    className="emoji-item relative group"
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div className="emoji-content">
                      <span className="text-xl bg-gray-100 p-1 rounded-md inline-block border border-gray-200 hover:bg-orange-50">
                        {model.emoji}
                      </span>
                      <div className="absolute -top-10 left-0 bg-black text-white p-1 rounded-sm opacity-0 group-hover:opacity-100 transition-opacity text-xs whitespace-nowrap">
                        Score: {model.probability.toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
          
          {/* Person 2 Response Section */}
          <div className="bg-white rounded-lg p-6 shadow-md border border-black">
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
              {animatedModels
                .filter(model => model.vote === 'ai')
                .map((model, index) => (
                  <div 
                    key={model.id} 
                    className="emoji-item relative group"
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div className="emoji-content">
                      <span className="text-xl bg-gray-100 p-1 rounded-md inline-block border border-gray-200 hover:bg-blue-50">
                        {model.emoji}
                      </span>
                      <div className="absolute -top-10 left-0 bg-black text-white p-1 rounded-sm opacity-0 group-hover:opacity-100 transition-opacity text-xs whitespace-nowrap">
                        Score: {model.probability.toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
        
        {/* Predict Button */}
        <div className="flex justify-center">
          <button
            onClick={predict}
            disabled={isLoading}
            className="bg-black hover:bg-gray-800 text-white font-medium py-2 px-8 rounded-full text-sm shadow-sm transition-colors relative min-w-[100px]">
            {isLoading ? (
              <div className="flex justify-center items-center">
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Judging...
              </div>
            ) : (
              "Judge!"
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

export default Index;