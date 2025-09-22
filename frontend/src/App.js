import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, NavLink } from 'react-router-dom';
import UploadForm from './components/UploadForm';
import SmartCheckForm from './components/SmartCheckForm';
import AIContentCheckForm from './components/AIContentCheckForm';

const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-br from-pink-200 via-purple-300 via-violet-400 to-indigo-500">
        {/* Navbar */}
        <nav className="sticky top-0 bg-white/90 backdrop-blur-sm shadow-md border-b border-gray-200 z-50">
          <div className="max-w-6xl mx-auto px-4 py-4 flex justify-between items-center">
            <div className="flex items-center gap-2">
              <img src="/logo.png" alt="Logo" className="h-8 w-8" />
              <span className="text-2xl font-bold text-indigo-900">TruthTrace AI</span>
              </div>
            <div className="space-x-6">
              <NavLink
                to="/ai-check"
                className={({ isActive }) =>
                  `text-sm font-semibold transition ${
                    isActive
                      ? 'text-indigo-900 border-b-2 border-indigo-700'
                      : 'text-gray-700 hover:text-indigo-900'
                  }`
                }
              >
                AI Detector
              </NavLink>
              <NavLink
                to="/smart-check"
                className={({ isActive }) =>
                  `text-sm font-semibold transition ${
                    isActive
                      ? 'text-indigo-900 border-b-2 border-indigo-700'
                      : 'text-gray-700 hover:text-indigo-900'
                  }`
                }
              >
                Plagiarism Check
              </NavLink>
              <NavLink
                to="/check"
                className={({ isActive }) =>
                  `text-sm font-semibold transition ${
                    isActive
                      ? 'text-indigo-900 border-b-2 border-indigo-700'
                      : 'text-gray-700 hover:text-indigo-900'
                  }`
                }
              >
                Two-Document Compare
              </NavLink>
            </div>
          </div>
        </nav>

        {/* Main Page Area */}
        <div className="max-w-4xl mx-auto py-10 px-4">
          <Routes>
            <Route path="/ai-check" element={<AIContentCheckForm />} />
            <Route path="/smart-check" element={<SmartCheckForm />} />
            <Route path="/check" element={<UploadForm />} />
            <Route path="/" element={<Navigate to="/check" />} />
          </Routes>
        </div>

      </div>
    </Router>
  );
};

export default App;
