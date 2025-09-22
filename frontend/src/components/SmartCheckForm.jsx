import React, { useState } from 'react';
import SimilarityDonut from './SimilarityDonut';

const SmartCheckForm = () => {
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

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/smart-check`, {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Error comparing document.');
      setResult(data);
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <div className="bg-white p-8 shadow-xl rounded-lg w-full max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-center text-indigo-900 mb-6">📚 Plagiarism Check</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-gray-700 mb-1">Upload Document:</label>
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
            {loading ? 'Checking...' : 'Check Document'}
          </button>
        </div>
      </form>

      {error && <p className="mt-4 text-center text-red-600 font-semibold">{error}</p>}

      {result && (
        <div className="mt-6 text-center">
          <h4 className="font-semibold text-lg">Similarity Score</h4>
          <SimilarityDonut similarity={result.similarity_score} />
        </div>
      )}
    </div>
  );
};

export default SmartCheckForm;
