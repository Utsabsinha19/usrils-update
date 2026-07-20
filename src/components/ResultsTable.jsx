import React from 'react';

export const ResultsTable = ({ data }) => {
  return (
    <div className="table-container">
      <table>
        <thead>
          <tr>
            <th>Method</th>
            <th>Accuracy</th>
            <th>Stability</th>
            <th>Generalization</th>
            <th>Cost</th>
            <th>CIM (Score)</th>
          </tr>
        </thead>
        <tbody>
          {data.map(r => (
            <tr key={r.method} className={r.highlight ? 'highlight' : ''}>
              <td>{r.method}</td>
              <td>{r.acc.toFixed(3)}</td>
              <td>{r.stab.toFixed(3)}</td>
              <td>{r.gen.toFixed(3)}</td>
              <td>{r.cost.toFixed(3)}</td>
              <td>{r.cim.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
