# Exercise 1 - Smoothing and Interpolation

# Input data and code hints
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)
plt.ion()

# Read a 3D volume (MRI scan of the brain) and extract the 2D slice (75) used in this exercises:
T1_file = 'IXI002-Guys-0828-T1.nii.gz'
T1_nib = nib.load( T1_file )
data = T1_nib.get_fdata()
img = data[:, :, 75]
T = np.flipud( img.T )

#plt.axis('off')
#plt.imshow(T, cmap='bone');

# Extract the 1D signal at the middle row of the slice:
row = round( T.shape[0] / 2 )
t = T[row, :] #pixel intensity signal at row y = 128
N = t.shape[0] # 256
x = np.arange(0, N)

# plot
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
fig.tight_layout(pad=5.0)

im = ax[0].imshow(T, cmap='bone')
ax[0].imshow(T, cmap='bone')
ax[0].hlines(row, xmin=1, xmax=255, alpha=0.6, color='r')
ax[0].set_xlabel('x pos')
ax[0].set_ylabel('y pos')
#fig.colorbar(im, ax=ax[0])

ax[1].scatter(x, t)
ax[1].set_xlabel('x pos')
ax[1].set_ylabel('pixel intensity (y={})'.format(row))

plt.show();

#%% Task 1: B-splines basis functions
"""
Implement a function that evaluates the uniform B-spline of orders 0, 1 and 3 at the locations provided in a vector x. 
This function should return a vector $\mathbf{y}$ that is the same length as $\mathbf{x}$.
"""
def eval_BSpline(x, order=0):
    """
    Evaluates the uniform B-spline of order "order" at the locations in vector "x"
    Order 0: box function
    Order 1: triangle function
    Order 3: cubic B-spline
    """
    x = np.array(x, dtype=float)
    
    #Initialize y (zeros vector)
    y = np.zeros_like(x)

    if order == 0:
        # if |x| < 0.5, B0(x) = 1, else B0 = 0
        y[np.abs(x) < 0.5] = 1.0

    elif order == 1:
        # if |x|<1, B1(x) = 1 - |x|, else B1 = 0
        y = np.maximum(1 - np.abs(x), 0)

    elif order == 3:
        # Cubic B-spline piecewise definition
        ax = np.abs(x)
        y[ax < 1] = 2/3-(ax[ax < 1])**2+(ax[ax < 1])**3/2
        y[(ax >= 1) & (ax < 2)] = (2 - ax[(ax >= 1) & (ax < 2)])**3 / 6
    else:
        raise ValueError("Only orders 0, 1, and 3 are supported.")
    return y

# Use `eval_BSpline` to plot the uniform B-spline of orders 0, 1 and 3 at locations: x = [-3.0,-2.99,-2.98, \ldots, 2.99,3.0]. 

# Test points
x = np.arange(-3.0, 3.01, 0.01)

plt.figure(figsize=(8,4))

for order in [0, 1, 3]:
    y = eval_BSpline(x, order)
    plt.plot(x, y, label=f"Order {order}")

plt.title("Uniform B-spline Basis Functions")
plt.xlabel("x position")
plt.ylabel("B-spline value")
plt.grid(True)
plt.legend()

plt.text(0.5, -0.25, "Figure 1", 
    ha='center', va='center', transform=plt.gca().transAxes)

plt.show()

#%% Task 2: 1D Smoothing
""""
Using the function `eval_BSpline` for evaluating a uniform B-spline of order 3, generate $M=6$ basis functions covering the entire domain x=0,1,..., 255
of the 1D signal t defined in the introduction (which has length N=256). The basis functions should be scaled by a factor 
h=(N-1)/(M-1), and shifted to be $h$ units apart (see book). Collect the obtained basis functions in a (NxM) matrix Phi$, and plot them.
"""
M = 6

# This scaling factor (51) ensures that the basis function 
# width matches the spacing needed to spread evenly across the full signal.
h = (N - 1) / (M - 1)

# Domain
x = np.arange(0,N)

"""
Build basis functions matrix Φ (NxM) where each column is a basis 
function evaluated at all points of the domain
"""
Phi = np.zeros((N, M)) # [256x6]

for m in range(M):
    """
    Note.
    d/h converts the x-coords into basis function units while subtracting 
    m centers each basis function at a different integer m in the scaled 
    coordinate system. Therefore, d/h - m effectively shifts the m-th 
    basis function so that it is centered at the right position along the 
    domain:
     m=0 → 1st basis function centered at 0
     m=1 → 2nd basis function centered at 1
     etc..
    """
    Phi[:, m] = eval_BSpline(x/h-m, order=3)

# Plot
plt.figure(figsize=(8, 4))
for m in range(M):
    plt.plot(x, Phi[:, m], label=f"Basis {m}")
plt.title("Cubic B-spline Basis Functions (M=6, N=256)")
plt.xlabel("x position")
plt.ylabel("Weight Value")
plt.legend(loc='upper right')
plt.grid(True)
plt.text(0.5, -0.2, "Figure 2", 
    ha='center', va='center', transform=plt.gca().transAxes)
plt.show()

""""
 Compute:
- the smoothing matrix: S = Phi @ Phi^T @ Phi^-1 @ Phi^T
- the smoothed signal:  t_smooth = St
Plots the results and the middle row of the smoothing matrix.  
"""

# Compute smoothing matrix S
A = Phi.T @ Phi # MxM

# Solve AX = Phi^T -> X = A^-1 Phi^T = (Phi.T @ Phi)^-1 @ Phi^T
X = np.linalg.solve(A, Phi.T) 
S = Phi @ X                   # smooth matrix
t_smooth = S @ t              # smoothed signal

# Plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
fig.tight_layout(pad=5.0)

# Left plot: original and smoothed signal
ax[0].plot(x, t, label="Original signal", alpha=0.6)
ax[0].plot(x, t_smooth, label="Smoothed signal", linewidth=2)
ax[0].set_title("Smoothing effect with Cubic B-spline Basis (M=6)")
ax[0].set_xlabel("x pos")
ax[0].set_ylabel("Signal Value")
ax[0].legend(loc='upper right')
ax[0].grid(True)

# Right plot: middle row of smoothing matrix
ax[1].plot(S[N//2, :], marker="x", markersize=4)
ax[1].set_title("Smoothing Weights at Signal Center Location")
ax[1].set_xlabel("x pos (column index)")
ax[1].set_ylabel("Weight Value")
ax[1].grid(True)

fig.text(0.5, 0.0, "Figure 3", ha='center', va='bottom', fontsize=16)
plt.show()

#%%
"""
Now repeat the same experiment when more basis functions are used: $M=16$ and $M=52$. Explain how the smoothing behavior change.
"""
# case M=16:
M = 16
h = (N - 1) / (M - 1)   # h = 17

# Build basis matrix Φ (N x M)
Phi = np.zeros((N, M)) # [256x16]

for m in range(M):
    Phi[:, m] = eval_BSpline(x/h-m, order=3)

# Compute smoothing matrix S
A = Phi.T @ Phi
X = np.linalg.solve(A, Phi.T)
S = Phi @ X
t_smooth = S @ t

# Plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
fig.tight_layout(pad=5.0)

# Left plot: original and smoothed signal
ax[0].plot(x, t, label="Original signal", alpha=0.6)
ax[0].plot(x, t_smooth, label="Smoothed signal", linewidth=2)
ax[0].set_title("Smoothing effect with Cubic B-spline Basis (M=16)")
ax[0].set_xlabel("x pos")
ax[0].set_ylabel("Signal Value")
ax[0].legend(loc='upper right')
ax[0].grid(True)

# Right plot: middle row of smoothing matrix
ax[1].plot(S[N//2, :], marker="x", markersize=4)
ax[1].set_title("Smoothing Weights at Signal Center Location")
ax[1].set_xlabel("x pos (column index)")
ax[1].set_ylabel("Weights Value")
ax[1].grid(True)

fig.text(0.5, 0.0, "Figure 4", ha='center', va='bottom', fontsize=16)
plt.show()

#%% case M=52:
M = 52
h = (N - 1) / (M - 1)   # h = 5

# Build basis matrix Φ (N x M)
Phi = np.zeros((N, M)) # [256x52]

for m in range(M):
    Phi[:, m] = eval_BSpline(x/h-m, order=3)

# Compute smoothing matrix S
A = Phi.T @ Phi
X = np.linalg.solve(A, Phi.T)
S = Phi @ X
t_smooth = S @ t  

# Plots
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
fig.tight_layout(pad=5.0)

# Left plot: original and smoothed signal
ax[0].plot(x, t, label="Original signal", alpha=0.6)
ax[0].plot(x, t_smooth, label="Smoothed signal", linewidth=2)
ax[0].set_title("Smoothing effect with Cubic B-spline Basis (M=52)")
ax[0].set_xlabel("x pos")
ax[0].set_ylabel("Signal Value")
ax[0].legend(loc='upper right')
ax[0].grid(True)

# Right plot: middle row of smoothing matrix
ax[1].plot(S[N//2, :], marker="x",markersize=4)
ax[1].set_title("Smoothing Weights at Signal Center Location")
ax[1].set_xlabel("x pos (column index)")
ax[1].set_ylabel("Weight Value")
ax[1].grid(True)

fig.text(0.5, 0.0, "Figure 5", ha='center', va='bottom', fontsize=16)
plt.show()

#%% Task 3: 1D Interpolation
"""
Implement B-spline interpolation of the 1D signal t, by evaluating the function y(x, w) at locations x = [120, 120.01, 120.02,..140]. 
Show your results, along with the corresponding part of the signal t$, for three different orders of B-splines: 0, 1 and 3.
"""
def bspline_interpolate(t, x_eval, order):
    """
    1D B-spline interpolation.
    t: array of samples (length N)
    x_eval: array of positions where to evaluate
    order: spline order (0=nearest, 1=linear, 3=cubic, …)
    """
    N = len(t)
    x_nodes = np.arange(N)

    # Build Phi at sample nodes
    Phi = np.zeros((N, N))
    for i in range(N):        # row = sample position
        for j in range(N):    # column = basis centered at j
            Phi[i, j] = eval_BSpline(x_nodes[i] - j, order)

    # Solve for coefficients
    if order <= 1:
        c = t.copy()   # for 0,1 coefficients == samples
    else:
        c = np.linalg.solve(Phi, t)

    # Evaluate at x_eval
    y_eval = np.zeros_like(x_eval, dtype=float)
    for i, x in enumerate(x_eval):
        for j in range(N):
            y_eval[i] += c[j] * eval_BSpline(x - j, order)

    return y_eval

x_eval = np.arange(120, 140, 0.01)

# Interpolation for orders 0,1,3
y0 = bspline_interpolate(t, x_eval, order=0)
y1 = bspline_interpolate(t, x_eval, order=1)
y3 = bspline_interpolate(t, x_eval, order=3)

# Plot results
plt.figure(figsize=(8,4))

# Original discrete samples in the interval
plt.plot(x[120:140], t[120:140], "o", label="Original samples")

plt.plot(x_eval, y0, label="Order 0 (nearest)", alpha=0.7)
plt.plot(x_eval, y1, label="Order 1 (linear)", alpha=0.7)
plt.plot(x_eval, y3, label="Order 3 (cubic)", alpha=0.7)

plt.title("B-spline Signal Interpolation (Orders 0,1,3)")
plt.xlabel("x position")
plt.ylabel("Signal Value")
plt.legend(loc="lower right")
plt.grid(True)
plt.figtext(0.5, -0.05, "Figure 6", ha="center", fontsize=12)
plt.show()

#%% Task 4: 2D Smoothing
"""
Now you are going to smooth the 2D image T, which has dimensions N1xN2=256x256, with M1=M2=5 B-spline-based 1D 
basis functions of order 3 for each direction. Use the Kronecker product to produce a (NxM) matrix Phi where N=N1N2 
and M = M1M2. 
Show a few 2D basis functions contained in the columns of $\boldsymbol{\Phi}$.
"""

from mpl_toolkits.mplot3d import Axes3D

N1 = N2 = N
M1 = M2 = 5
h1 = (N1 - 1) / (M1 - 1)   # spacing in dim 1
h2 = (N2 - 1) / (M2 - 1)   # spacing in dim 2

x1 = np.arange(N1)
x2 = np.arange(N2)

# Initializie 1D basis matrices
Phi1 = np.zeros((N1, M1))
Phi2 = np.zeros((N2, M2))

# Build the matrices
for m in range(M1):
    Phi1[:, m] = eval_BSpline(x1 / h1 - m, order=3)

for m in range(M2):
    Phi2[:, m] = eval_BSpline(x2 / h2 - m, order=3)

# Kronecker product: 2D basis matrix
Phi = np.kron(Phi2, Phi1)   # shape (N1*N2, M1*M2)

# Visualize a few 2D basis functions
cols_to_plot = [0, 6, 12, 18, 24]
n_cols = len(cols_to_plot)

fig = plt.figure(figsize=(12, 6))

for idx, i in enumerate(cols_to_plot):
    
    # Determine row and column for 3+2 layout
    if idx < 3:  # first 3 plots (row 1)
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')
    else:        # last 2 plots (row 2)
        ax = fig.add_subplot(2, 3, idx+1, projection='3d')

    basis_col = Phi[:, i].reshape(N1, N2, order="F")
    X, Y = np.meshgrid(np.arange(N2), np.arange(N1))
    ax.plot_wireframe(X, Y, basis_col, rstride=16, cstride=16)
    ax.set_title(f"Basis {i}")

plt.subplots_adjust(hspace=0.3)
plt.show()
#%%
"""
Now:
- vectorize the 2D image $\mathbf{T}$ into a 1D signal $\mathbf{t}$
- smooth using $\boldsymbol{\Phi}$
- re-arrange the resulting 1D signal $\mathbf{\hat{t}}$ back into a 2D image $\mathbf{\hat{T}}$
- show the result.
"""

# Vectorize the 2D image
t = T.reshape(-1, 1, order='F') # [N1*N2, 1]

# Compute weights
# Solve w = (Phi^T Phi)^(-1) Phi^T t
w = np.linalg.solve(Phi.T @ Phi, Phi.T @ t)

# Smooth the vector t
t_smooth = Phi @ w   

# Reshape back to 2D
T_smooth = t_smooth.reshape(N1, N2, order='F')

# Display results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(T, cmap='bone')
plt.title("Original Image")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(T_smooth, cmap='bone')
plt.title("Smoothed Image")
plt.colorbar()

plt.suptitle("Figure 7", y=0.01, fontsize=12)
plt.show()
#%%
"""
Also perform the same smoothing operation by exploiting the separability of the 2D basis functions, i.e, 
perform row-wise and then column-wise smoothing instead. Show the results, and verify that they are identical 
(use np.allclose()).
"""
# Compute intermediate matrices for separable smoothing
A1 = Phi1.T @ Phi1
A2 = Phi2.T @ Phi2

# Compute weight matrix W
temp = np.linalg.solve(A1, Phi1.T @ T @ Phi2)
W = np.linalg.solve(A2, temp.T).T  # [M1, M2]

# Smoothed image
T_smooth_sep = Phi1 @ W @ Phi2.T  # [N1, N2]

# Display results
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(T_smooth, cmap='bone')
plt.title("Kronecker Product Smoothing")
plt.colorbar()

plt.subplot(1,2,2)
plt.imshow(T_smooth_sep, cmap='bone')
plt.title("Separable Row/Column Smoothing")
plt.colorbar()

plt.suptitle("Figure 8", y=0.01, fontsize=12)
plt.show()

# Verify that they are identical
print("Are the two methods identical?", np.allclose(T_smooth, T_smooth_sep))

#%% Task 5: 2D Interpolation
"""
For B-spline order 0, 1 and 3, compute the weights $\mathbf{W}$ of the 2D interpolation model fitted to T by 
exploiting the separability of its basis functions. Then show, as an image, interpolated function values y(x,w)
for x=(x1, x2)T with:
- x1=120.0, 120.1, 120.2,...,130.0
- x2=120.0, 120.1, 120.2,...,130.0
"""

# evaluation points
x_eval = np.arange(120.0, 130.01, 0.1)
n_eval = len(x_eval)

x1 = np.arange(N1)
x2 = np.arange(N2)

# Orders to test
orders = [0, 1, 3]

fig, axes = plt.subplots(1, len(orders), figsize=(15,5))

for idx, order in enumerate(orders):

    # For each spline order, initialize the basis matrices
    Phi1 = np.zeros((N1, N1))
    Phi2 = np.zeros((N2, N2))

    # Build the matrices
    for i in range(N1):
        Phi1[:, i] = eval_BSpline(x1 - i, order)
    for i in range(N2):
        Phi2[:, i] = eval_BSpline(x2 - i, order)

    # Compute 2D weights using separability
    A1 = Phi1.T @ Phi1
    A2 = Phi2.T @ Phi2
    
    temp = np.linalg.solve(A1, Phi1.T @ T @ Phi2)
    W = np.linalg.solve(A2, temp.T).T

    # Compute 1D basis matrices at evaluation points
    Phi1_eval = np.zeros((n_eval, N1))
    Phi2_eval = np.zeros((n_eval, N2))
    for i in range(N1):
        Phi1_eval[:, i] = eval_BSpline(x_eval - i, order)
    for i in range(N2):
        Phi2_eval[:, i] = eval_BSpline(x_eval - i, order)

    # Compute interpolated image
    y_eval = Phi1_eval @ W @ Phi2_eval.T  # size: n_eval x n_eval

    # Display in corresponding subplot
    ax = axes[idx]
    im = ax.imshow(y_eval, extent=(x_eval[0], x_eval[-1], x_eval[-1], x_eval[0]),
                   origin='upper', cmap='bone')
    ax.set_title(f"Order {order}")
    ax.set_xlabel("x2")
    ax.set_ylabel("x1")

plt.suptitle("Figure 9", y=-0.1, fontsize=16)
plt.tight_layout()

plt.show()
