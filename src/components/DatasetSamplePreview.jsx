import React from 'react';
import { Table, CheckCircle } from 'lucide-react';

export const DatasetSamplePreview = () => {
  const sampleRows = [
    { age: 39, workclass: "State-gov", eduNum: 13, marital: "Never-married", occupation: "Adm-clerical", capGain: 2174, hpw: 40, income: 0 },
    { age: 50, workclass: "Self-emp-not-inc", eduNum: 13, marital: "Married-civ-spouse", occupation: "Exec-managerial", capGain: 0, hpw: 13, income: 0 },
    { age: 38, workclass: "Private", eduNum: 9, marital: "Divorced", occupation: "Handlers-cleaners", capGain: 0, hpw: 40, income: 0 },
    { age: 53, workclass: "Private", eduNum: 7, marital: "Married-civ-spouse", occupation: "Handlers-cleaners", capGain: 0, hpw: 40, income: 0 },
    { age: 28, workclass: "Private", eduNum: 13, marital: "Married-civ-spouse", occupation: "Prof-specialty", capGain: 0, hpw: 40, income: 1 }
  ];

  return (
    <div className="module-card" style={{ gap: '1rem', marginTop: '1.5rem' }}>
      <div className="module-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Table className="logo-icon" style={{ width: '20px', height: '20px', color: 'var(--secondary)' }} />
          <h2 className="module-title">Sample Tabular Dataset Preview (UCI-Adult)</h2>
        </div>
        <span className="badge" style={{ backgroundColor: 'rgba(0, 245, 212, 0.1)', color: 'var(--secondary)' }}>Standard Input Format</span>
      </div>

      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', lineHeight: '1.4' }}>
        This sample shows the format of a typical tabular dataset processed by the USRILS pipeline. 
        During <strong>Stage S1 (Normalization)</strong>, numeric columns are scaled via min-max normalization, 
        and categorical columns (like <em>Workclass</em> or <em>Occupation</em>) are automatically mapped to unique 
        integers (Label Encoding) to form the feature vector $X$.
      </p>

      <div style={{ overflowX: 'auto', border: '1px solid var(--glass-border)', borderRadius: '8px' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.75rem', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid var(--glass-border)' }}>
              <th style={{ padding: '0.6rem 0.8rem' }}>Age</th>
              <th style={{ padding: '0.6rem 0.8rem' }}>Workclass</th>
              <th style={{ padding: '0.6rem 0.8rem' }}>Education-Num</th>
              <th style={{ padding: '0.6rem 0.8rem' }}>Marital-Status</th>
              <th style={{ padding: '0.6rem 0.8rem' }}>Occupation</th>
              <th style={{ padding: '0.6rem 0.8rem' }}>Capital-Gain</th>
              <th style={{ padding: '0.6rem 0.8rem' }}>Hours-per-Week</th>
              <th style={{ padding: '0.6rem 0.8rem', color: 'var(--secondary)' }}>Income (Target Y)</th>
            </tr>
          </thead>
          <tbody>
            {sampleRows.map((row, idx) => (
              <tr key={idx} style={{ borderBottom: idx < sampleRows.length - 1 ? '1px solid rgba(255,255,255,0.05)' : 'none' }}>
                <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'var(--font-mono)' }}>{row.age}</td>
                <td style={{ padding: '0.6rem 0.8rem', color: 'var(--text-muted)' }}>{row.workclass}</td>
                <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'var(--font-mono)' }}>{row.eduNum}</td>
                <td style={{ padding: '0.6rem 0.8rem', color: 'var(--text-muted)' }}>{row.marital}</td>
                <td style={{ padding: '0.6rem 0.8rem', color: 'var(--text-muted)' }}>{row.occupation}</td>
                <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'var(--font-mono)' }}>{row.capGain}</td>
                <td style={{ padding: '0.6rem 0.8rem', fontFamily: 'var(--font-mono)' }}>{row.hpw}</td>
                <td style={{ padding: '0.6rem 0.8rem', fontWeight: 'bold', color: row.income === 1 ? 'var(--primary)' : 'var(--text-muted)' }}>
                  {row.income === 1 ? '>50K (1)' : '<=50K (0)'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', background: 'rgba(0, 245, 212, 0.02)', border: '1px solid rgba(0, 245, 212, 0.1)', padding: '0.6rem', borderRadius: '6px', fontSize: '0.7rem' }}>
        <CheckCircle size={14} color="var(--secondary)" />
        <span><strong>Pipeline Integration:</strong> Uploading any CSV/JSON matching this format to the Custom Trainer will trigger automatic feature processing, normalization, and training.</span>
      </div>
    </div>
  );
};
