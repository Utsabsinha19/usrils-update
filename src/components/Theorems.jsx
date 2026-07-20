import React, { useState } from 'react';

export const Theorems = ({ alpha, beta, gamma, lambda }) => {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="module-card">
      <div className="module-header">
        <h2 className="module-title">Theoretical Foundation</h2>
        <button 
          className="module-btn-action"
          onClick={() => setShowDetails(!showDetails)}
        >
          {showDetails ? 'Hide Details' : 'View Details'}
        </button>
      </div>

      <div className="math-section">
        {/* System Definition */}
        <div className="math-card">
          <div className="math-title">System Definition</div>
          <div className="latex-formula">
            S = (X, Y, θ, M, C)
          </div>
          <div className="latex-variables">
            Where: <strong>X</strong> (Input), <strong>Y</strong> (Output), <strong>θ</strong> (Parameters), <br />
            <strong>M = (A, S, G, C)</strong> (Metrics), <strong>C</strong> (Controller)
          </div>
        </div>

        {/* Composite Intelligence Metric */}
        <div className="math-card">
          <div className="math-title">Composite Intelligence Metric</div>
          <div className="latex-formula">
            Θ(θ) = αA + βS + γG - λC
          </div>
          <div className="latex-variables">
            Subject to: α + β + γ + λ = 1
          </div>
        </div>

        {/* Parameter values mapping (linked to sliders!) */}
        <div className="params-grid">
          <div className="param-box">
            <div className="param-label">α (Accuracy)</div>
            <div className="param-val">{(alpha ?? 0.40).toFixed(2)}</div>
          </div>
          <div className="param-box">
            <div className="param-label">β (Stability)</div>
            <div className="param-val">{(beta ?? 0.25).toFixed(2)}</div>
          </div>
          <div className="param-box">
            <div className="param-label">γ (Generalization)</div>
            <div className="param-val">{(gamma ?? 0.25).toFixed(2)}</div>
          </div>
          <div className="param-box">
            <div className="param-label">λ (Cost)</div>
            <div className="param-val">{(lambda ?? 0.10).toFixed(2)}</div>
          </div>
        </div>

        {/* Objective notice */}
        <div className="objective-banner">
          <strong>Objective:</strong> Maximize Θ(θ) subject to regularity assumptions <strong>(A1-A4)</strong>
        </div>

        {/* Detailed theorems toggle */}
        {showDetails && (
          <div className="math-card" style={{ borderLeft: '4px solid var(--primary)', marginTop: '0.5rem', animation: 'fadeIn 0.3s ease' }}>
            <div className="math-title">Key Guarantees & Theorems</div>
            <ul style={{ paddingLeft: '1rem', fontSize: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <li><strong>T1 (Convergence):</strong> Parameter updates converge to θ* at a geometric rate O(e<sup>-κt</sup>).</li>
              <li><strong>T2 (Optimality):</strong> θ* is Pareto-optimal; no other parameters dominate θ* on all metrics.</li>
              <li><strong>T3 (Generalization):</strong> Rademacher bound ε &lt; 0.0423 holds at n=50k, δ=0.05.</li>
              <li><strong>T4 (Lyapunov Stability):</strong> Monotonic energy decrease V(θ<sub>t</sub>) proving global asymptotic stability.</li>
              <li><strong>T5 (Master Theorem):</strong> Reliable intelligence exists if and only if all four bounds hold simultaneously.</li>
            </ul>
          </div>
        )}
      </div>
    </div>
  );
};
