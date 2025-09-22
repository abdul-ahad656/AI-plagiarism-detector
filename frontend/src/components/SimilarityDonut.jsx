import React from 'react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const SimilarityDonut = ({ similarity }) => {
  const percentage = (similarity * 100).toFixed(2);
  const value = parseFloat(percentage);

  const color =
    value > 80 ? '#ef4444' : value > 50 ? '#eab308' : '#22c55e';

  const message =
    value > 80
      ? '⚠️ High similarity — Possible plagiarism'
      : value > 50
      ? '⚠️ Moderate similarity — Needs review'
      : '✅ Low similarity — Likely original';

  return (
    <div className="mt-6 text-center">
      <div className="w-40 h-40 mx-auto animate-fade-in">
        <CircularProgressbar
          value={value}
          text={`${percentage}%`}
          styles={buildStyles({
            textColor: '#1f2937',
            pathColor: color,
            trailColor: '#e5e7eb',
            textSize: '16px',
            pathTransitionDuration: 0.6,
          })}
        />
      </div>
      <p className="mt-4 font-semibold text-gray-700">{message}</p>
    </div>
  );
};

export default SimilarityDonut;
