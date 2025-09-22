import React, { useState } from 'react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const AIContentCheckForm = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError('');
    setLoading(true);

    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${process.env.REACT_APP_API_URL}/detect-ai`, {
      method: 'POST',
      body: formData,
    });

    try {
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error.');
        setResult(data);
      } catch (err) {
          setError("Could not read server response. Please try again.");
      }

    setLoading(false);
  };

  return (
    <div className="bg-white p-8 shadow-xl rounded-lg w-full max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-center text-indigo-900 mb-6">🤖 AI Content Detector</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-gray-700 mb-1">Upload a Document:</label>
          <input
            type="file"
            accept=".txt,.pdf,.docx"
            onChange={(e) => setFile(e.target.files[0])}
            required
            className="w-full border border-gray-300 px-4 py-2 rounded"
          />
        </div>

        <div className="text-center">
          <button
            type="submit"
            disabled={loading}
            className={`${
              loading ? 'bg-gray-400' : 'bg-indigo-900 hover:bg-indigo-700'
            } text-white px-6 py-2 rounded transition flex items-center justify-center gap-2`}
          >
            {loading && (
              <span className="inline-block animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></span>
            )}
            {loading ? 'Checking...' : 'Detect AI Content'}
          </button>
        </div>
      </form>

      {error && <p className="mt-4 text-center text-red-600 font-semibold">{error}</p>}

      {result && (
  <div className="mt-6 text-center animate-fade-in">
    <div className="w-40 h-40 mx-auto">
      <CircularProgressbar
        value={result.ai_score * 100}
        text={`${(result.ai_score * 100).toFixed(2)}% AI`}
        styles={buildStyles({
          textColor: '#1f2937',
          pathColor:
            result.ai_score > 0.7
              ? '#ef4444'
              : result.ai_score > 0.4
              ? '#eab308'
              : '#22c55e',
          trailColor: '#e5e7eb',
          textSize: '16px',
          pathTransitionDuration: 0.6,
        })}
      />
    </div>
    <p className="mt-4 font-semibold text-gray-700">{result.verdict}</p>
  </div>
)}
    </div>
  );
};

export default AIContentCheckForm;
