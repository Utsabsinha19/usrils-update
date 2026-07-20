import React, { useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import { useSimulation } from '../hooks/useSimulation';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

export const Simulator = ({
  alpha, setAlpha,
  beta, setBeta,
  gamma, setGamma,
  lambda, setLambda,
  target, setTarget,
  gain, setGain
}) => {
  const { lrHistory, cimHistory, finalLr, finalBranch, isSimulating, runSimulation } = useSimulation();

  useEffect(() => {
    if (!isSimulating) {
      runSimulation(alpha, beta, gamma, lambda, target, gain, false);
    }
  }, [alpha, beta, gamma, lambda, target, gain, runSimulation, isSimulating]);

  const handleSliderChange = (type, val) => {
    const value = parseFloat(val);
    if (type === 'alpha') {
      setAlpha(value);
      const remainder = 1.0 - value;
      const sumOthers = beta + gamma + lambda;
      if (sumOthers > 0) {
        setBeta(beta * remainder / sumOthers);
        setGamma(gamma * remainder / sumOthers);
        setLambda(lambda * remainder / sumOthers);
      } else {
        setBeta(remainder / 3);
        setGamma(remainder / 3);
        setLambda(remainder / 3);
      }
    } else if (type === 'beta') {
      setBeta(value);
      const remainder = 1.0 - value;
      const sumOthers = alpha + gamma + lambda;
      if (sumOthers > 0) {
        setAlpha(alpha * remainder / sumOthers);
        setGamma(gamma * remainder / sumOthers);
        setLambda(lambda * remainder / sumOthers);
      } else {
        setAlpha(remainder / 3);
        setGamma(remainder / 3);
        setLambda(remainder / 3);
      }
    } else if (type === 'gamma') {
      setGamma(value);
      const remainder = 1.0 - value;
      const sumOthers = alpha + beta + lambda;
      if (sumOthers > 0) {
        setAlpha(alpha * remainder / sumOthers);
        setBeta(beta * remainder / sumOthers);
        setLambda(lambda * remainder / sumOthers);
      } else {
        setAlpha(remainder / 3);
        setBeta(remainder / 3);
        setLambda(remainder / 3);
      }
    } else if (type === 'lambda') {
      setLambda(value);
      const remainder = 1.0 - value;
      const sumOthers = alpha + beta + gamma;
      if (sumOthers > 0) {
        setAlpha(alpha * remainder / sumOthers);
        setBeta(beta * remainder / sumOthers);
        setGamma(gamma * remainder / sumOthers);
      } else {
        setAlpha(remainder / 3);
        setBeta(remainder / 3);
        setGamma(remainder / 3);
      }
    }
  };

  const steps = Array.from({ length: 50 }, (_, i) => i + 1);

  const chartData = {
    labels: steps,
    datasets: [
      {
        label: 'Learning Rate (eta_t)',
        data: lrHistory,
        borderColor: '#00f5d4', // Cyan
        backgroundColor: 'transparent',
        borderWidth: 2,
        yAxisID: 'y',
        pointRadius: 0,
        tension: 0.1
      },
      {
        label: 'Composite Score (Theta_t)',
        data: cimHistory,
        borderColor: '#9d4edd', // Purple
        backgroundColor: 'transparent',
        borderWidth: 2,
        yAxisID: 'y1',
        pointRadius: 0,
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: { color: '#f3f0f7', font: { family: 'Inter', size: 10 } }
      }
    },
    scales: {
      x: {
        grid: { color: 'rgba(255, 255, 255, 0.03)' },
        ticks: { color: '#6b7280', font: { family: 'JetBrains Mono', size: 9 } }
      },
      y: {
        position: 'left',
        grid: { color: 'rgba(255, 255, 255, 0.03)' },
        ticks: { color: '#00f5d4', font: { family: 'JetBrains Mono', size: 9 } },
        title: { display: false }
      },
      y1: {
        position: 'right',
        grid: { display: false },
        ticks: { color: '#9d4edd', font: { family: 'JetBrains Mono', size: 9 } },
        title: { display: false }
      }
    }
  };

  return (
    <div className="module-card">
      <div className="module-header">
        <h2 className="module-title">Closed-Loop Regulation Simulator</h2>
        <span className="badge" style={{ backgroundColor: 'rgba(0, 245, 212, 0.1)', color: '#00f5d4', borderColor: 'rgba(0, 245, 212, 0.3)' }}>● Live</span>
      </div>

      <div className="simulator-layout">
        {/* Sliders Panel */}
        <div className="sim-controls">
          <div className="sim-sliders-card">
            <div className="sim-slider-box">
              <div className="sim-slider-lbls">
                <span className="sim-slider-lbl">Alpha (Accuracy Weight)</span>
                <span className="sim-slider-val">{alpha.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.05" 
                value={alpha} 
                onChange={(e) => handleSliderChange('alpha', e.target.value)} 
                disabled={isSimulating}
              />
            </div>

            <div className="sim-slider-box">
              <div className="sim-slider-lbls">
                <span className="sim-slider-lbl">Beta (Stability Weight)</span>
                <span className="sim-slider-val">{beta.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.05" 
                value={beta} 
                onChange={(e) => handleSliderChange('beta', e.target.value)} 
                disabled={isSimulating}
              />
            </div>

            <div className="sim-slider-box">
              <div className="sim-slider-lbls">
                <span className="sim-slider-lbl">Gamma (Generalization Weight)</span>
                <span className="sim-slider-val">{gamma.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.05" 
                value={gamma} 
                onChange={(e) => handleSliderChange('gamma', e.target.value)} 
                disabled={isSimulating}
              />
            </div>

            <div className="sim-slider-box">
              <div className="sim-slider-lbls">
                <span className="sim-slider-lbl">Lambda (Cost Weight)</span>
                <span className="sim-slider-val">{lambda.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="1" 
                step="0.05" 
                value={lambda} 
                onChange={(e) => handleSliderChange('lambda', e.target.value)} 
                disabled={isSimulating}
              />
            </div>

            <div className="sim-slider-box">
              <div className="sim-slider-lbls">
                <span className="sim-slider-lbl">Regulation Gain (μ_r)</span>
                <span className="sim-slider-val">{gain.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0.05" 
                max="0.30" 
                step="0.01" 
                value={gain} 
                onChange={(e) => setGain(parseFloat(e.target.value))} 
                disabled={isSimulating}
              />
            </div>

            <div className="sim-slider-box">
              <div className="sim-slider-lbls">
                <span className="sim-slider-lbl">Target CIM (Θ*)</span>
                <span className="sim-slider-val">{target.toFixed(2)}</span>
              </div>
              <input 
                type="range" 
                min="0.5" 
                max="0.95" 
                step="0.01" 
                value={target} 
                onChange={(e) => setTarget(parseFloat(e.target.value))} 
                disabled={isSimulating}
              />
            </div>
          </div>

          <button 
            className="sim-btn-engage" 
            onClick={() => runSimulation(alpha, beta, gamma, lambda, target, gain, true)}
            disabled={isSimulating}
            style={{
              opacity: isSimulating ? 0.75 : 1,
              cursor: isSimulating ? 'not-allowed' : 'pointer'
            }}
          >
            {isSimulating ? 'Regulating Step Size...' : 'Engage Regulation'}
          </button>
        </div>

        {/* Chart & Stats Panel */}
        <div className="sim-chart-section">
          <div className="sim-chart-wrapper">
            <Line data={chartData} options={chartOptions} />
          </div>
          
          <div className="sim-status-bar">
            <div>
              Controller State: <span className={`sim-status-val ${finalBranch.toLowerCase()}`}>{finalBranch}</span>
            </div>
            <div>
              Final Learning Rate: <span className="sim-status-val" style={{ color: 'var(--secondary)' }}>{finalLr.toFixed(3)}</span>
            </div>
            <div>
              Steps Logged: <span className="sim-status-val" style={{ color: 'var(--text-main)' }}>50</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
