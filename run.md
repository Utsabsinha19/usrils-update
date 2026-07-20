npm run dev --prefix frontend

cd frontend
npm run dev

.venv\Scripts\python run_experiments.py

.venv\Scripts\python -m pytest

PS C:\Users\Utsab Sinha\Downloads\usrils> .venv\Scripts\python run_experiments.py
Using device: cpu

==========================================
Running experiments on: ADULT
==========================================

Run 1/3...
  SGD: CIM = 0.6559
  Adam: CIM = 0.7227
  AdaGrad: CIM = 0.7149
  MTL: CIM = 0.7181
  CO: CIM = 0.4892
  USRILS: CIM = nan

Run 2/3...
  SGD: CIM = 0.6635
  Adam: CIM = 0.7187
  AdaGrad: CIM = 0.7140
  MTL: CIM = 0.7110
  CO: CIM = 0.4942
  USRILS: CIM = nan

Run 3/3...
  SGD: CIM = 0.6446
  Adam: CIM = 0.7233
  AdaGrad: CIM = 0.7097
  MTL: CIM = 0.7084
  CO: CIM = 0.4863
  USRILS: CIM = nan

Final results table for ADULT:
 Method   Acc  Stab   Gen  Cost   CIM CIM_std
    SGD 0.860 0.578 0.984 0.800 0.655   0.008
   Adam 0.959 0.680 0.993 0.800 0.722   0.002
AdaGrad 0.941 0.664 1.002 0.800 0.713   0.002
    MTL 0.918 0.698 1.004 0.800 0.712   0.004
     CO 0.307 0.787 1.001 0.800 0.490   0.003
 USRILS   NaN   NaN   NaN 0.819   NaN     NaN

Statistical paired t-tests (CIM compared to USRILS):
  USRILS vs SGD: t-stat = nan, p-value = nan -> p = nan (Not Significant)
  USRILS vs Adam: t-stat = nan, p-value = nan -> p = nan (Not Significant)
  USRILS vs AdaGrad: t-stat = nan, p-value = nan -> p = nan (Not Significant)
  USRILS vs MTL: t-stat = nan, p-value = nan -> p = nan (Not Significant)
  USRILS vs CO: t-stat = nan, p-value = nan -> p = nan (Not Significant)

==========================================
Running experiments on: MNIST
==========================================

Run 1/3...
  SGD: CIM = -0.3662
  Adam: CIM = -0.4047
  AdaGrad: CIM = -0.4324
  MTL: CIM = 71.6792
  CO: CIM = -0.2030
  USRILS: CIM = -0.3004

Run 2/3...
  SGD: CIM = -0.3686
  Adam: CIM = -0.4068
  AdaGrad: CIM = -0.4140
  MTL: CIM = 70.6477
  CO: CIM = -0.2036
  USRILS: CIM = -0.2947

Run 3/3...
  SGD: CIM = -0.3642
  Adam: CIM = -0.4047
  AdaGrad: CIM = -0.4078
  MTL: CIM = 70.9052
  CO: CIM = -0.1812
  USRILS: CIM = -0.2930

Final results table for MNIST:
 Method     Acc  Stab     Gen  Cost    CIM CIM_std
    SGD  -1.316 0.051   0.909 0.800 -0.366   0.002
   Adam  -1.336 0.045   0.792 0.800 -0.405   0.001
AdaGrad  -1.331 0.078   0.699 0.800 -0.418   0.010
    MTL -22.857 0.000 321.200 0.800 71.077   0.438
     CO  -1.303 0.620   1.001 0.800 -0.196   0.010
 USRILS  -1.290 0.039   1.166 0.813 -0.296   0.003

Statistical paired t-tests (CIM compared to USRILS):
  USRILS vs SGD: t-stat = 29.7080, p-value = 1.13e-03 -> p = 0.0011 (Not Significant)
  USRILS vs Adam: t-stat = 43.2323, p-value = 5.35e-04 -> p < 0.001 (Significant)
  USRILS vs AdaGrad: t-stat = 23.7253, p-value = 1.77e-03 -> p = 0.0018 (Not Significant)
  USRILS vs MTL: t-stat = -228.7850, p-value = 1.91e-05 -> p < 0.001 (Significant)
  USRILS vs CO: t-stat = -16.3320, p-value = 3.73e-03 -> p = 0.0037 (Not Significant)

==========================================
Running experiments on: CIFAR
==========================================

Run 1/3...
  SGD: CIM = -0.2473
  Adam: CIM = -0.3082
  AdaGrad: CIM = -0.3172
  MTL: CIM = 66.1530
  CO: CIM = -0.1714
  USRILS: CIM = -0.1252

Run 2/3...
  SGD: CIM = -0.2402
  Adam: CIM = -0.3149
  AdaGrad: CIM = -0.3029
  MTL: CIM = 33.2385
  CO: CIM = -0.1730
  USRILS: CIM = -0.1304

Run 3/3...
  SGD: CIM = -0.2455
  Adam: CIM = -0.2991
  AdaGrad: CIM = -0.3456
  MTL: CIM = 55.2264
  CO: CIM = -0.1748
  USRILS: CIM = -0.1309

Final results table for CIFAR:
 Method      Acc  Stab     Gen  Cost    CIM CIM_std
    SGD   -1.302 0.437   1.002 0.833 -0.244   0.003
   Adam   -1.314 0.290   0.917 0.833 -0.307   0.006
AdaGrad   -1.309 0.261   0.878 0.833 -0.322   0.018
    MTL -120.098 0.000 398.648 0.833 51.539  13.688
     CO   -1.303 0.724   1.001 0.833 -0.173   0.001
 USRILS   -1.304 0.632   1.001 0.157 -0.129   0.003

Statistical paired t-tests (CIM compared to USRILS):
  USRILS vs SGD: t-stat = 32.3709, p-value = 9.53e-04 -> p < 0.001 (Significant)
  USRILS vs Adam: t-stat = 34.1907, p-value = 8.54e-04 -> p < 0.001 (Significant)
  USRILS vs AdaGrad: t-stat = 15.8245, p-value = 3.97e-03 -> p = 0.0040 (Not Significant)
  USRILS vs MTL: t-stat = -5.3390, p-value = 3.33e-02 -> p = 0.0333 (Not Significant)
  USRILS vs CO: t-stat = 42.4886, p-value = 5.53e-04 -> p < 0.001 (Significant)
PS C:\Users\Utsab Sinha\Downloads\usrils> .venv\Scripts\python -m pytest
================================================ test session starts ================================================
platform win32 -- Python 3.10.0, pytest-7.4.3, pluggy-1.6.0
rootdir: C:\Users\Utsab Sinha\Downloads\usrils
plugins: anyio-3.7.1, asyncio-0.21.1, typeguard-4.5.1
asyncio: mode=strict
collected 5 items                                                                                                     

tests\test_usrils.py .....                                                                                     [100%] 

================================================= 5 passed in 2.33s ================================================



