#test Broyden method for solving nonlinear equations

import numpy as np

A = np.array([[8.0, 1.0],
              [4.0, 3.0]])          # 初始矩阵：正好是 x0=(1,0) 处的 Jacobian

def F(x):
    """x 为 (2,1) 列向量，返回 (2,1) 列向量"""
    x1, x2 = x[0, 0], x[1, 0]
    return np.array([[4*x1**2 + x2**2 + 2*x1*x2 - x2 - 2],
                     [2*x1**2 + 3*x1*x2 + x2**2 - 3]])

x0 = np.array([[1.0],
               [0.0]])              # 关键：必须是列向量，不能用 Python list

for k in range(20):
    r = -np.linalg.inv(A) @ F(x0)           # 牛顿步：解 A r = -F(x0)
    x1 = x0 + r
    y = F(x1) - F(x0)
    A = A + (y - A @ r) @ r.T / (r.T @ r)   # good Broyden 更新
    print(f"iter {k}: x = {x1.ravel()}")
    x0 = x1
    if np.max(np.abs(r)) < 0.5e-5:
        break



