import React from 'react';
import { Radar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  RadialLinearScale,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend
);

export const RadarChart = ({ data }) => {
  const labels = ['Accuracy', 'Stability', 'Generalization', '1 - Cost', 'CIM'];
  
  const datasets = data.map(r => {
    const isUsrils = r.method === 'USRILS';
    const colorHex = isUsrils ? '#00f5d4' : (r.method === 'Adam' ? '#9d4edd' : '#a39bb4');
    
    return {
      label: r.method,
      data: [r.acc, r.stab, r.gen, 1.0 - r.cost, r.cim],
      borderColor: colorHex,
      backgroundColor: isUsrils ? 'rgba(0, 245, 212, 0.1)' : 'rgba(255, 255, 255, 0.02)',
      borderWidth: isUsrils ? 3 : 1.5,
      pointRadius: isUsrils ? 4 : 2,
    };
  });

  const chartData = { labels, datasets };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#f3f0f7',
          font: {
            family: 'Inter',
            size: 11
          }
        }
      },
      tooltip: {
        backgroundColor: 'rgba(8, 3, 18, 0.95)',
        titleColor: '#00f5d4',
        bodyColor: '#f3f0f7',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        borderWidth: 1
      }
    },
    scales: {
      r: {
        grid: {
          color: 'rgba(255, 255, 255, 0.08)'
        },
        angleLines: {
          color: 'rgba(255, 255, 255, 0.08)'
        },
        pointLabels: {
          color: '#a39bb4',
          font: {
            family: 'Inter',
            size: 11
          }
        },
        ticks: {
          backdropColor: 'transparent',
          color: '#a39bb4',
          z: 10
        },
        min: 0,
        max: 1.0
      }
    }
  };

  return (
    <div className="chart-wrapper">
      <Radar data={chartData} options={options} />
    </div>
  );
};
