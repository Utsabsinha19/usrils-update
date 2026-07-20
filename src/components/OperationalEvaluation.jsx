import React from 'react';

const evaluationConfigs = [
  { id: 1, dataset: 'UCI-Adult', n: '48,842', model: 'XGBoost', kappa: '0.867', epsilon: '0.0412', stability: '0.0071', score: '0.918' },
  { id: 2, dataset: 'MNIST', n: '60,000', model: 'CNN', kappa: '0.892', epsilon: '0.0389', stability: '0.0059', score: '0.936' },
  { id: 3, dataset: 'CIFAR-10', n: '50,000', model: 'ResNet-18', kappa: '0.871', epsilon: '0.0423', stability: '0.0066', score: '0.923' }
];

export const OperationalEvaluation = () => {
  return (
    <div className="operational-table-card">
      <div className="module-header">
        <h2 className="module-title" style={{ borderLeft: '4px solid var(--primary)', paddingLeft: '0.75rem' }}>
          Operational Evaluation (Selected Configurations)
        </h2>
        <button className="module-btn-action">View All Tables</button>
      </div>

      <div className="table-container">
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Dataset</th>
              <th>n (Samples)</th>
              <th>Model</th>
              <th>κ (Rate)</th>
              <th>ε (Bound)</th>
              <th>Stability (V_50)</th>
              <th>CIM Score</th>
              <th>Result</th>
            </tr>
          </thead>
          <tbody>
            {evaluationConfigs.map(cfg => (
              <tr key={cfg.id}>
                <td>{cfg.id}</td>
                <td>{cfg.dataset}</td>
                <td>{cfg.n}</td>
                <td>{cfg.model}</td>
                <td>{cfg.kappa}</td>
                <td>{cfg.epsilon}</td>
                <td>{cfg.stability}</td>
                <td>{cfg.score}</td>
                <td>
                  <span className="badge-pass">✓ Pass</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
