import numpy as np
import matplotlib.pyplot as plt

def plot_lines(augmented_matrix):
    """
    Plot linear equations from an augmented matrix [A|b]
    
    Parameters:
    augmented_matrix: numpy array of shape (n, n+1) where each row represents
                      a linear equation in the form: a1*x1 + a2*x2 + ... = b
    """
    # Extract the number of variables (columns - 1)
    n_vars = augmented_matrix.shape[1] - 1
    
    if n_vars != 2:
        print("Plotting only supports 2D systems")
        return
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot each line
    x = np.linspace(-10, 10, 400)
    
    for i, row in enumerate(augmented_matrix):
        a1, a2, b = row[0], row[1], row[2]
        
        # Solve for y: a1*x + a2*y = b  =>  y = (b - a1*x) / a2
        if a2 != 0:
            y = (b - a1 * x) / a2
            plt.plot(x, y, label=f'Equation {i+1}', linewidth=2)
        else:
            # Vertical line: a1*x = b  =>  x = b/a1
            if a1 != 0:
                x_val = b / a1
                plt.axvline(x=x_val, label=f'Equation {i+1}', linewidth=2)
    
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('System of Linear Equations')
    plt.show()
