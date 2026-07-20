import { useState, useCallback, useRef } from 'react';

export const useSimulation = () => {
    const [lrHistory, setLrHistory] = useState([]);
    const [cimHistory, setCimHistory] = useState([]);
    const [finalLr, setFinalLr] = useState(0.05);
    const [finalBranch, setFinalBranch] = useState('HOLD');
    const [isSimulating, setIsSimulating] = useState(false);

    const intervalRef = useRef(null);

    const runSimulation = useCallback((alpha, beta, gamma, lambda, target, gain, animate = false) => {
        // Clear any active running simulation
        if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
        }

        const steps = 50;
        const tempLrHistory = [];
        const tempCimHistory = [];
        const branches = [];
        
        let lr = 0.050;
        let cim = 0.500;
        let branch = 'HOLD';

        for (let t = 1; t <= steps; t++) {
            tempLrHistory.push(lr);
            tempCimHistory.push(cim);
            branches.push(branch);

            // Simulate parameter changes converging
            const targetAcc = 0.90 + 0.09 * Math.sin(t / 10);
            const targetStab = 0.85 + 0.10 * Math.cos(t / 5);
            const targetGen = 0.80 + 0.10 * Math.sin(t / 15);
            const targetCost = 0.70 + 0.05 * Math.sin(t / 20);

            const newCim = alpha * targetAcc + beta * targetStab + gamma * targetGen - lambda * targetCost;
            
            const delta_theta = newCim - target;
            const epsilon_ctrl = 0.02;

            if (delta_theta > epsilon_ctrl) {
                lr = Math.min(lr * (1 + gain * Math.abs(delta_theta)), 0.500);
                branch = 'EXPAND';
            } else if (Math.abs(delta_theta) <= epsilon_ctrl) {
                branch = 'HOLD';
            } else {
                lr = Math.max(lr * (1 - gain * Math.abs(delta_theta)), 0.001);
                branch = 'CONTRACT';
            }
            
            cim = cim + 0.15 * (newCim - cim);
        }

        if (!animate) {
            setIsSimulating(false);
            setLrHistory(tempLrHistory);
            setCimHistory(tempCimHistory);
            setFinalLr(tempLrHistory[steps - 1]);
            setFinalBranch(branches[steps - 1]);
        } else {
            setIsSimulating(true);
            setLrHistory([]);
            setCimHistory([]);
            setFinalLr(0.05);
            setFinalBranch('HOLD');

            let currentStep = 0;
            intervalRef.current = setInterval(() => {
                currentStep++;
                if (currentStep <= steps) {
                    setLrHistory(tempLrHistory.slice(0, currentStep));
                    setCimHistory(tempCimHistory.slice(0, currentStep));
                    setFinalLr(tempLrHistory[currentStep - 1]);
                    setFinalBranch(branches[currentStep - 1]);
                } else {
                    clearInterval(intervalRef.current);
                    intervalRef.current = null;
                    setIsSimulating(false);
                }
            }, 30); // 30ms * 50 steps = 1.5 seconds simulation runtime
        }
    }, []);

    return {
        lrHistory,
        cimHistory,
        finalLr,
        finalBranch,
        isSimulating,
        runSimulation
    };
};
