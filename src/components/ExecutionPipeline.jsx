import React from 'react';

const pipelineStages = [
  { id: 1, name: 'S1 Normalise', desc: 'Raw input → x_hat' },
  { id: 2, name: 'S2 Extract', desc: 'x_hat → z ∈ R^k' },
  { id: 3, name: 'S3 Predict', desc: 'z → y_hat, L(θ_t)' },
  { id: 4, name: 'S4 Evaluate', desc: 'Metrics: A_t, S_t, G_t, C_t' },
  { id: 5, name: 'S5 Regulate', desc: 'μ_t → θ_t+1 (Expand / Hold / Contract)' },
  { id: 6, name: 'S6 Update', desc: 'θ_t+1 → Metrics & θ_t+1' }
];

export const ExecutionPipeline = () => {
  return (
    <div className="module-card">
      <div className="module-header">
        <h2 className="module-title">Execution Pipeline</h2>
        <span className="badge">6 Stages</span>
      </div>

      <div className="pipeline-layout">
        <div className="pipeline-steps">
          {pipelineStages.map(stage => (
            <div key={stage.id} className="pipeline-step-item">
              <div className="step-num-badge">{stage.id}</div>
              <div className="step-details">
                <span className="step-title">{stage.name}</span>
                <span className="step-formula">{stage.desc}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="pipeline-visual">
          <svg className="pipeline-svg" viewBox="0 0 200 250">
            {/* Connection path */}
            <path
              d="M 50,30 L 150,70 L 50,110 L 150,150 L 50,190 L 150,230"
              fill="none"
              stroke="rgba(157, 78, 221, 0.25)"
              strokeWidth="2"
              strokeDasharray="4 4"
            />
            {/* Winding nodes */}
            <g>
              <circle cx="50" cy="30" r="12" fill="#080312" stroke="#00f5d4" strokeWidth="2" />
              <text x="50" y="34" textAnchor="middle" fill="#00f5d4" fontSize="10" fontWeight="bold">1</text>

              <circle cx="150" cy="70" r="12" fill="#080312" stroke="#9d4edd" strokeWidth="2" />
              <text x="150" y="74" textAnchor="middle" fill="#9d4edd" fontSize="10" fontWeight="bold">2</text>

              <circle cx="50" cy="110" r="12" fill="#080312" stroke="#f15bb5" strokeWidth="2" />
              <text x="50" y="114" textAnchor="middle" fill="#f15bb5" fontSize="10" fontWeight="bold">3</text>

              <circle cx="150" cy="150" r="12" fill="#080312" stroke="#34d399" strokeWidth="2" />
              <text x="150" y="154" textAnchor="middle" fill="#34d399" fontSize="10" fontWeight="bold">4</text>

              <circle cx="50" cy="190" r="12" fill="#080312" stroke="#fbbf24" strokeWidth="2" />
              <text x="50" y="194" textAnchor="middle" fill="#fbbf24" fontSize="10" fontWeight="bold">5</text>

              <circle cx="150" cy="230" r="12" fill="#080312" stroke="#00f5d4" strokeWidth="2" />
              <text x="150" y="234" textAnchor="middle" fill="#00f5d4" fontSize="10" fontWeight="bold">6</text>
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
};
