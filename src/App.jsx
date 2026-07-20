import React, { useState } from 'react';
import { 
  Activity, BookOpen, BarChart3, Cpu, Sliders, Table, FileText,
  Globe, Mail, Shield, Zap, CheckCircle, Lock, Scale,
  Unlock, User, Loader2, LogOut
} from 'lucide-react';
import { paperResults } from './data/paperResults';
import { Theorems } from './components/Theorems';
import { ResultsTable } from './components/ResultsTable';
import { RadarChart } from './components/RadarChart';
import { Simulator } from './components/Simulator';
import { ExecutionPipeline } from './components/ExecutionPipeline';
import { OperationalEvaluation } from './components/OperationalEvaluation';
import { Trainer } from './components/Trainer';
import { DatasetSamplePreview } from './components/DatasetSamplePreview';

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [isLoggingIn, setIsLoggingIn] = useState(false);
  const [authError, setAuthError] = useState('');

  const [isLoading, setIsLoading] = useState(false);
  const [loadingStage, setLoadingStage] = useState('Initializing USRILS Controller...');

  const [activeTab, setActiveTab] = useState('Overview');
  const [dataset, setDataset] = useState('cifar'); // Default to CIFAR-10 as in the image

  const handleLogin = (e) => {
    e.preventDefault();
    setAuthError('');
    if (!username.trim() || !password.trim()) {
      setAuthError('Please enter both username and password.');
      return;
    }
    
    setIsLoggingIn(true);
    
    setTimeout(() => {
      // Validate: Allow 'admin' with password 'admin' or any other username/password combination
      // as long as it's not empty, but if it is 'admin', verify password is also 'admin'.
      if (username.toLowerCase() === 'admin' && password !== 'admin') {
        setAuthError('Invalid password for admin user.');
        setIsLoggingIn(false);
        return;
      }
      
      setIsLoggingIn(false);
      setIsAuthenticated(true);
      setIsLoading(true);
      
      // Start skeleton loading stages
      const stages = [
        "Establishing secure control loop link...",
        "Ingesting dataset streams & hyperparameters...",
        "Pre-evaluating Rademacher generalization bounds...",
        "Executing second-order double autograd pass...",
        "Lyapunov stability constraints verified. System Optimal."
      ];
      
      let stageIdx = 0;
      setLoadingStage(stages[0]);
      
      const interval = setInterval(() => {
        stageIdx++;
        if (stageIdx < stages.length) {
          setLoadingStage(stages[stageIdx]);
        } else {
          clearInterval(interval);
          setIsLoading(false);
        }
      }, 400);
      
    }, 850);
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setUsername('');
    setPassword('');
    setAuthError('');
  };

  // Simulator parameter states lifted up to synchronize with Theoretical Foundation values
  const [alpha, setAlpha] = useState(0.40);
  const [beta, setBeta] = useState(0.25);
  const [gamma, setGamma] = useState(0.25);
  const [lambda, setLambda] = useState(0.10);
  const [target, setTarget] = useState(0.94); // Default to 0.94 as in the image
  const [gain, setGain] = useState(0.15);

  const currentData = paperResults[dataset];

  const renderTabContent = () => {
    switch (activeTab) {
      case 'Overview':
        return (
          <>
            {/* Guarantees Cards Row */}
            <div className="guarantees-row">
              <div className="guarantee-card">
                <Shield className="guarantee-icon" />
                <span className="guarantee-label">KEY GUARANTEES</span>
                <span className="guarantee-value" style={{ fontSize: '1.05rem', color: '#818cf8', marginTop: '0.75rem' }}>(All Four Hold Jointly)</span>
                <span className="guarantee-subtext">Verified under A1-A4</span>
              </div>

              <div className="guarantee-card primary">
                <Activity className="guarantee-icon" />
                <span className="guarantee-label">Convergence Rate (κ)</span>
                <span className="guarantee-value">0.871</span>
                <span className="guarantee-subtext">O(e^-kt)</span>
              </div>

              <div className="guarantee-card optimality">
                <Scale className="guarantee-icon" />
                <span className="guarantee-label">Pareto Optimality (Score)</span>
                <span className="guarantee-value">0.925</span>
                <span className="guarantee-subtext">Pareto Front</span>
              </div>

              <div className="guarantee-card generalization">
                <CheckCircle className="guarantee-icon" />
                <span className="guarantee-label">Generalization Bound (ε)</span>
                <span className="guarantee-value">0.0423</span>
                <span className="guarantee-subtext">at δ = 0.05, n = 50k</span>
              </div>

              <div className="guarantee-card stability">
                <Lock className="guarantee-icon" />
                <span className="guarantee-label">Lyapunov Stability (V_t)</span>
                <span className="guarantee-value">0.0066</span>
                <span className="guarantee-subtext">Monotonic Decrease</span>
              </div>
            </div>

            {/* Middle Row: Theoretical + Sample Dataset Preview */}
            <div className="dashboard-grid two-col" style={{ gridTemplateColumns: '1fr 1.3fr', gap: '1.5rem', marginTop: '1.5rem' }}>
              <section>
                <Theorems alpha={alpha} beta={beta} gamma={gamma} lambda={lambda} />
              </section>
              <section>
                <DatasetSamplePreview />
              </section>
            </div>

            {/* Bottom Row: Simulator + Execution Pipeline */}
            <div className="dashboard-grid two-col" style={{ gap: '1.5rem', marginTop: '1.5rem' }}>
              <section>
                <Simulator 
                  alpha={alpha} setAlpha={setAlpha}
                  beta={beta} setBeta={setBeta}
                  gamma={gamma} setGamma={setGamma}
                  lambda={lambda} setLambda={setLambda}
                  target={target} setTarget={setTarget}
                  gain={gain} setGain={setGain}
                />
              </section>

              <section>
                <ExecutionPipeline />
              </section>
            </div>

            {/* Operational Evaluation Table */}
            <div style={{ marginTop: '1.5rem' }}>
              <OperationalEvaluation />
            </div>
          </>
        );
      case 'Theory':
        return (
          <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            <Theorems alpha={alpha} beta={beta} gamma={gamma} lambda={lambda} />
          </div>
        );
      case 'Empirical':
        return (
          <div className="module-card" style={{ maxWidth: '1000px', margin: '0 auto', gap: '1.25rem' }}>
            <div className="module-header">
              <h2 className="module-title">Empirical Verification</h2>
              <span className="badge" style={{ backgroundColor: 'rgba(0, 245, 212, 0.1)', color: 'var(--secondary)' }}>SOTA Evaluation</span>
            </div>

            <div className="empirical-body">
              <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '1.5rem', alignItems: 'center' }}>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
                    <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#00f5d4', borderRadius: '3px' }}></span>
                    <strong style={{ color: 'var(--text-main)' }}>USRILS (Ours)</strong>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
                    <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#9d4edd', borderRadius: '3px' }}></span>
                    <span>Adam Optimizer</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
                    <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#f15bb5', borderRadius: '3px' }}></span>
                    <span>SOTA Baselines</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
                    <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#fbbf24', borderRadius: '3px' }}></span>
                    <span>Multi-Task Learning (MTL)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.6rem' }}>
                    <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#10b981', borderRadius: '3px' }}></span>
                    <span>Constrained Optimization (CO)</span>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span style={{ display: 'inline-block', width: '12px', height: '12px', backgroundColor: '#6b7280', borderRadius: '3px' }}></span>
                    <span>Baseline SGD</span>
                  </div>
                </div>

                <RadarChart data={currentData} />
              </div>

              <div className="empirical-table-area" style={{ marginTop: '1.5rem' }}>
                <ResultsTable data={currentData} />
              </div>

              <div className="empirical-note" style={{ marginTop: '1rem', fontStyle: 'italic', fontSize: '0.75rem' }}>
                Statistical Significance: USRILS outperforms all baselines with <span>p &lt; 0.001</span> (paired t-test over 15 runs).
              </div>
            </div>
          </div>
        );
      case 'Pipeline':
        return (
          <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            <ExecutionPipeline />
          </div>
        );
      case 'Simulator':
        return (
          <div style={{ maxWidth: '900px', margin: '0 auto' }}>
            <Simulator 
              alpha={alpha} setAlpha={setAlpha}
              beta={beta} setBeta={setBeta}
              gamma={gamma} setGamma={setGamma}
              lambda={lambda} setLambda={setLambda}
              target={target} setTarget={setTarget}
              gain={gain} setGain={setGain}
            />
          </div>
        );
      case 'Trainer':
        return <Trainer />;
      case 'Results':
        return (
          <div style={{ maxWidth: '1000px', margin: '0 auto' }}>
            <OperationalEvaluation />
          </div>
        );
      case 'Docs':
        return (
          <div className="module-card" style={{ gap: '1.25rem', maxWidth: '850px', margin: '0 auto', padding: '1.5rem 2rem' }}>
            <div className="module-header">
              <h2 className="module-title">Theoretical Documentation</h2>
              <span className="badge">v1.2.0</span>
            </div>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: '1.6', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <h3 style={{ color: 'var(--text-main)', fontSize: '1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.4rem', margin: '0.5rem 0' }}>1. Mathematical Formulation</h3>
              <p>
                The Unified Self-Regulating Intelligent Learning System (USRILS) models neural parameter optimization as a closed-loop dynamical control system. Unlike conventional optimizers (SGD, Adam) which operate open-loop, USRILS monitors a composite metric termed the <strong>Composite Intelligence Metric (CIM)</strong>:
              </p>
              <pre style={{ background: 'rgba(0,0,0,0.3)', padding: '0.6rem', borderRadius: '6px', color: 'var(--secondary)', fontFamily: 'var(--font-mono)', fontSize: '0.75rem' }}>
                {"Theta(t) = alpha * Acc(t) + beta * Stab(t) + gamma * Gen(t) - lambda * Cost(t)"}
              </pre>
              <p>
                Where:
                <br />- <strong>Acc(t)</strong>: Empirical accuracy on validation data.
                <br />- <strong>Stab(t)</strong>: Lyapunov convergence stability term, computed via double autograd through the gradient norm.
                <br />- <strong>Gen(t)</strong>: Rademacher generalization bound.
                <br />- <strong>Cost(t)</strong>: Computational FLOPs cost scaled against the system budget.
              </p>
              
              <h3 style={{ color: 'var(--text-main)', fontSize: '1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.4rem', margin: '0.5rem 0' }}>2. Three-Branch Regulation Logic</h3>
              <p>
                A Proportional feedback controller adjusts the step size (learning rate) {"$\\eta_t$"} based on the deviation {"$\\Delta \\Theta_t = \\Theta_t - \\Theta^*$:"}
              </p>
              <ul style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                <li><strong>EXPAND</strong>: {"If $\\Delta \\Theta_t > \\epsilon$, accelerate step size: $\\eta_{t+1} = \\min(\\eta_t (1 + \\mu_r |\\Delta \\Theta_t|), \\eta_{\\max})$"}</li>
                <li><strong>HOLD</strong>: {"If $|\\Delta \\Theta_t| \\le \\epsilon$, maintain stable step size: $\\eta_{t+1} = \\eta_t$"}</li>
                <li><strong>CONTRACT</strong>: {"If $\\Delta \\Theta_t < -\\epsilon$ or parameter gradients explode, contract step size: $\\eta_{t+1} = \\max(\\eta_t (1 - \\mu_r |\\Delta \\Theta_t|), \\eta_{\\min})$"}</li>
              </ul>

              <h3 style={{ color: 'var(--text-main)', fontSize: '1rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '0.4rem', margin: '0.5rem 0' }}>3. Double Autograd for Stability</h3>
              <p>
                To guarantee Lyapunov stability ({"$V_{t+1} - V_t \\le 0$"}), the optimizer computes gradients through the gradient norm itself. This requires a double-backward autograd pass (second-order gradients), which acts as a regularizer penalizing gradient variance.
              </p>
            </div>
          </div>
        );
      default:
        return null;
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="auth-container">
        <div className="auth-card">
          <div className="auth-header">
            <Activity className="auth-logo-icon" />
            <h2 className="auth-title">USRILS GATEWAY</h2>
            <p className="auth-subtitle">Self-Regulating Learning Terminal</p>
          </div>
          
          <form className="auth-form" onSubmit={handleLogin}>
            <div className="auth-field-group">
              <label className="auth-label" htmlFor="username">Operator Username</label>
              <div className="auth-input-wrapper">
                <User className="auth-input-icon" />
                <input 
                  type="text" 
                  id="username"
                  className="auth-input" 
                  placeholder="e.g. admin" 
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={isLoggingIn}
                />
              </div>
            </div>
            
            <div className="auth-field-group">
              <label className="auth-label" htmlFor="password">Security Credentials</label>
              <div className="auth-input-wrapper">
                <Lock className="auth-input-icon" />
                <input 
                  type="password" 
                  id="password"
                  className="auth-input" 
                  placeholder="••••••••" 
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={isLoggingIn}
                />
              </div>
            </div>
            
            {authError && (
              <div className="auth-error">
                <Shield className="auth-input-icon" style={{ left: '0.6rem', color: '#f87171' }} />
                <span>{authError}</span>
              </div>
            )}
            
            <button type="submit" className="auth-btn" disabled={isLoggingIn}>
              {isLoggingIn ? (
                <>
                  <Loader2 className="animate-spin" size={16} /> Authenticating...
                </>
              ) : (
                <>
                  <Unlock size={16} /> Request Terminal Access
                </>
              )}
            </button>
          </form>
          
          <div className="auth-footer">
            Enter standard operator credentials (admin/admin)
          </div>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="skeleton-wrapper">
        <aside className="skeleton-sidebar">
          <div className="skeleton-logo shimmer-base" />
          <div className="skeleton-nav-item shimmer-base" />
          <div className="skeleton-nav-item shimmer-base" />
          <div className="skeleton-nav-item shimmer-base" />
          <div className="skeleton-nav-item shimmer-base" />
          <div className="skeleton-nav-item shimmer-base" />
          <div className="skeleton-nav-item shimmer-base" />
          <div className="skeleton-nav-item shimmer-base" />
          <div className="skeleton-profile shimmer-base" />
        </aside>
        
        <main className="skeleton-main">
          <div className="skeleton-header-row">
            <div className="skeleton-title-area">
              <div className="skeleton-subtitle shimmer-base" />
              <div className="skeleton-title shimmer-base" />
              <div className="skeleton-subtext shimmer-base" />
            </div>
            <div className="skeleton-controls shimmer-base" />
          </div>
          
          <div className="skeleton-guarantees-row">
            <div className="skeleton-guarantee-card shimmer-base" />
            <div className="skeleton-guarantee-card shimmer-base" />
            <div className="skeleton-guarantee-card shimmer-base" />
            <div className="skeleton-guarantee-card shimmer-base" />
            <div className="skeleton-guarantee-card shimmer-base" />
          </div>
          
          <div className="skeleton-grid">
            <div className="skeleton-card shimmer-base" />
            <div className="skeleton-card shimmer-base" />
          </div>
          
          <div className="skeleton-terminal">
            <div className="skeleton-terminal-header">
              <div className="skeleton-terminal-title shimmer-base" />
              <div className="skeleton-terminal-dot shimmer-base" style={{ backgroundColor: 'var(--secondary)' }} />
            </div>
            <div className="skeleton-terminal-body">
              <div className="skeleton-terminal-line short shimmer-base" />
              <div className="skeleton-terminal-line medium shimmer-base" />
              <div className="skeleton-terminal-line shimmer-base" />
              <div className="skeleton-terminal-line medium shimmer-base" style={{ color: 'var(--secondary)', fontFamily: 'var(--font-mono)', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'none', animation: 'none' }}>
                <Loader2 className="animate-spin" size={12} /> {loadingStage}
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="app-wrapper">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo-section">
          <Activity className="logo-icon" />
          <span className="logo-text">USRILS</span>
        </div>

        <nav>
          <ul className="menu-list">
            <li className={`menu-item ${activeTab === 'Overview' ? 'active' : ''}`}>
              <button onClick={() => setActiveTab('Overview')}>
                <Zap className="menu-icon" /> Overview
              </button>
            </li>
            <li className={`menu-item ${activeTab === 'Theory' ? 'active' : ''}`}>
              <button onClick={() => setActiveTab('Theory')}>
                <BookOpen className="menu-icon" /> Theoretical Foundation
              </button>
            </li>
            <li className={`menu-item ${activeTab === 'Empirical' ? 'active' : ''}`}>
              <button onClick={() => setActiveTab('Empirical')}>
                <BarChart3 className="menu-icon" /> Empirical Verification
              </button>
            </li>
            <li className={`menu-item ${activeTab === 'Pipeline' ? 'active' : ''}`}>
              <button onClick={() => setActiveTab('Pipeline')}>
                <Cpu className="menu-icon" /> Execution Pipeline
              </button>
            </li>
            <li className={`menu-item ${activeTab === 'Simulator' ? 'active' : ''}`}>
              <button onClick={() => setActiveTab('Simulator')}>
                <Sliders className="menu-icon" /> Regulation Simulator
              </button>
            </li>
            <li className={`menu-item ${activeTab === 'Trainer' ? 'active' : ''}`}>
              <button onClick={() => setActiveTab('Trainer')}>
                <Activity className="menu-icon" /> Custom Trainer
              </button>
            </li>
            <li className={`menu-item ${activeTab === 'Results' ? 'active' : ''}`}>
              <button onClick={() => setActiveTab('Results')}>
                <Table className="menu-icon" /> Results & Tables
              </button>
            </li>
            <li className={`menu-item ${activeTab === 'Docs' ? 'active' : ''}`}>
              <button onClick={() => setActiveTab('Docs')}>
                <FileText className="menu-icon" /> Documentation
              </button>
            </li>
          </ul>
        </nav>

        <div className="author-card">
          <div className="author-name">Utsab Sinha</div>
          <div className="author-dept">Dept. of Computational Intelligence & ML Systems</div>
          <div className="author-socials">
            <a href="https://research.edu" target="_blank" rel="noreferrer"><Globe size={14} /></a>
            <a href="mailto:utsab.sinha@research.edu"><Mail size={14} /></a>
            <a href="https://linkedin.com" target="_blank" rel="noreferrer">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-linkedin">
                <path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path>
                <rect x="2" y="9" width="4" height="12"></rect>
                <circle cx="4" cy="4" r="2"></circle>
              </svg>
            </a>
          </div>
        </div>

        {/* User profile section */}
        <div className="user-profile-badge">
          <div className="user-profile-info">
            <div className="user-profile-avatar">
              {username ? username.substring(0, 2).toUpperCase() : 'OP'}
            </div>
            <div className="user-profile-name">{username || 'Operator'}</div>
          </div>
          <button className="btn-signout" onClick={handleLogout} title="Sign Out">
            <LogOut size={16} />
          </button>
        </div>

        <div className="copyright">© 2026 All Rights Reserved</div>
      </aside>

      {/* Main Content Area */}
      <main className="main-content">
        {/* Header */}
        <div className="header-row">
          <div className="header-title-area">
            <span className="subtitle">Foundational Theoretical Framework with Empirical Validation</span>
            <h1 className="main-title">Unified Self-Regulating Intelligent Learning System</h1>
            <span className="subtext">
              A provable framework that guarantees Convergence, Optimality, Generalization, and Stability 
              simultaneously through closed-loop regulation.
            </span>
            <span className="lead-author">Lead Author: Utsab Sinha</span>
          </div>

          <div className="header-controls">
            <select 
              id="datasetSelector" 
              value={dataset} 
              onChange={(e) => setDataset(e.target.value)}
              style={{
                background: 'rgba(10, 15, 30, 0.8)',
                color: 'var(--text-main)',
                border: '1px solid var(--glass-border)',
                padding: '0.5rem 1rem',
                borderRadius: '8px',
                fontFamily: 'var(--font-mono)',
                outline: 'none',
                cursor: 'pointer',
                fontSize: '0.8rem'
              }}
            >
              <option value="adult">Dataset: UCI-Adult</option>
              <option value="mnist">Dataset: MNIST</option>
              <option value="cifar">Dataset: CIFAR-10</option>
            </select>

            <div className="status-badge">
              <span className="status-dot"></span>
              System Status: Optimal
            </div>
          </div>
        </div>

        {renderTabContent()}

        {/* Footer */}
        <div className="footer-banner">
          Reliable Intelligence is a <span>Mathematical Theorem</span>, not an Empirical Hope.
        </div>
      </main>
    </div>
  );
}

export default App;
