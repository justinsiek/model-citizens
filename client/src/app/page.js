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
    <div className='flex flex-col justify-center items-center h-screen p-4'>
      <div className='w-full max-w-4xl mb-8'>
        <h3 className='text-xl font-semibold mb-4 text-center'>Prompt: {prompt}</h3>
      </div>
      
      <div className='w-full max-w-4xl flex flex-row gap-6 mb-8'>
        {/* User Response Section - Left Side */}
        <div className='flex-1 flex flex-col'>
          <h4 className='text-lg font-medium mb-2'>Your Response:</h4>
          <textarea
            className='border border-gray-400 p-2 rounded-lg mb-2 h-48 w-full'
            value={userResponse}
            onChange={(e) => setUserResponse(e.target.value)}
            placeholder="Enter your response..."
          />
          <div className='flex flex-wrap gap-1 min-h-10'>
            {results.map((probability, index) => (
              probability >= 0.5 ? 
                <span key={index} className='text-2xl' title={`Probability: ${probability.toFixed(2)}`}>
                  {personEmojis[index % personEmojis.length]}
                </span> : null
            ))}
          </div>
        </div>
        
        {/* AI Response Section - Right Side */}
        <div className='flex-1 flex flex-col'>
          <h4 className='text-lg font-medium mb-2'>AI Response:</h4>
          <div className='border border-gray-400 p-2 rounded-lg mb-2 h-48 w-full overflow-auto'>
            {aiResponse}
          </div>
          <div className='flex flex-wrap gap-1 min-h-10'>
            {results.map((probability, index) => (
              probability < 0.5 ? 
                <span key={index} className='text-2xl' title={`Probability: ${probability.toFixed(2)}`}>
                  {personEmojis[index % personEmojis.length]}
                </span> : null
            ))}
          </div>
        </div>
      </div>
      
      <button
        className='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-10 rounded'
        onClick={predict}>
        Predict!
      </button>
    </div>
  );
}

export default Index;