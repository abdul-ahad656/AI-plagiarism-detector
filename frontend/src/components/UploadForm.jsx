import React, { useState } from 'react';
import SimilarityDonut from './SimilarityDonut';

const UploadForm = () => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setResult(null);
    setError('');
    setLoading(true);

    const formData = new FormData();
    formData.append('file1', file1);
    formData.append('file2', file2);

    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/check`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) throw new Error('Error comparing files.');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    }

    setLoading(false);
  };

  return (
    <div className="bg-white p-8 shadow-xl rounded-lg w-full max-w-xl mx-auto">
      <h2 className="text-2xl font-bold text-center text-indigo-900 mb-6">📄 Compare Two Documents</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-gray-700 mb-1">Upload Document 1:</label>
          <input type="file" accept=".txt,.pdf,.docx" onChange={(e) => setFile1(e.target.files[0])} required
            className="w-full border border-gray-300 px-4 py-2 rounded" />
        </div>

        <div>
          <label className="block text-gray-700 mb-1">Upload Document 2:</label>
          <input type="file" accept=".txt,.pdf,.docx" onChange={(e) => setFile2(e.target.files[0])} required
            className="w-full border border-gray-300 px-4 py-2 rounded" />
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
            {loading ? 'Checking...' : 'Check Similarity'}
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

export default UploadForm;
