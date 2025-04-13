'use client'

import React, { useState } from 'react';

function Index() {
  const [prompt, setPrompt] = useState('Do you like pinapple on pizza?');
  const [aiResponse, setAiResponse] = useState('yes I love pizza with pinapple');
  const [userResponse, setUserResponse] = useState('');
  const [result, setResult] = useState('');

  const predict = () => {
    fetch(`http://127.0.0.1:5000/api/predict?user_response=${userResponse}&prompt=${prompt}&ai_response=${aiResponse}`)
      .then(response => {
        return response.json();
      })
      .then(data => {
        setResult(`Result: ${data.result}`);
      });
  };

  return (
    <div className='flex flex-col justify-center items-center h-screen'>
      <p className='mb-4'>Prompt: {prompt}</p>  
      <p className='mb-4'>AI Response: {aiResponse}</p>

      <input
        className='border border-gray-400 p-2 w-1/4 rounded-lg mb-4'
        type="text"
        value={userResponse}
        onChange={(e) => setUserResponse(e.target.value)}
        placeholder="Enter text..."
      />
      <button
        className='bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-10 rounded' 
        onClick={predict}>Predict!</button>
      <p className='mb-4'>{result}</p>
      
    </div>
  );
}

export default Index;