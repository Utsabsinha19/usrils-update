export const paperResults = {
    adult: [
        { method: 'SGD', acc: 0.847, stab: 0.721, gen: 0.831, cost: 0.350, cim: 0.792 },
        { method: 'Adam', acc: 0.861, stab: 0.754, gen: 0.843, cost: 0.420, cim: 0.810 },
        { method: 'AdaGrad', acc: 0.855, stab: 0.742, gen: 0.837, cost: 0.400, cim: 0.803 },
        { method: 'MTL', acc: 0.863, stab: 0.761, gen: 0.849, cost: 0.460, cim: 0.814 },
        { method: 'CO', acc: 0.859, stab: 0.758, gen: 0.845, cost: 0.440, cim: 0.809 },
        { method: 'USRILS', acc: 0.912, stab: 0.871, gen: 0.903, cost: 0.720, cim: 0.864, highlight: true }
    ],
    mnist: [
        { method: 'SGD', acc: 0.981, stab: 0.843, gen: 0.977, cost: 0.310, cim: 0.925 },
        { method: 'Adam', acc: 0.987, stab: 0.872, gen: 0.983, cost: 0.380, cim: 0.939 },
        { method: 'AdaGrad', acc: 0.983, stab: 0.850, gen: 0.980, cost: 0.330, cim: 0.930 },
        { method: 'MTL', acc: 0.988, stab: 0.865, gen: 0.984, cost: 0.410, cim: 0.941 },
        { method: 'CO', acc: 0.985, stab: 0.875, gen: 0.982, cost: 0.390, cim: 0.938 },
        { method: 'USRILS', acc: 0.993, stab: 0.941, gen: 0.989, cost: 0.750, cim: 0.960, highlight: true }
    ],
    cifar: [
        { method: 'SGD', acc: 0.912, stab: 0.791, gen: 0.905, cost: 0.360, cim: 0.876 },
        { method: 'Adam', acc: 0.931, stab: 0.821, gen: 0.923, cost: 0.450, cim: 0.897 },
        { method: 'AdaGrad', acc: 0.915, stab: 0.802, gen: 0.910, cost: 0.400, cim: 0.881 },
        { method: 'MTL', acc: 0.934, stab: 0.835, gen: 0.926, cost: 0.480, cim: 0.902 },
        { method: 'CO', acc: 0.928, stab: 0.840, gen: 0.921, cost: 0.460, cim: 0.899 },
        { method: 'USRILS', acc: 0.951, stab: 0.923, gen: 0.944, cost: 0.810, cim: 0.923, highlight: true }
    ]
};
