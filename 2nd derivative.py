import numpy as np
import matplotlib.pyplot as plt

def W(x, a=0.5, b=3, N=200):
    total = 0
    for n in range(N):
        total += a**n * np.cos((b**n) * np.pi * x)
    return total

def dW_dx(x, a=0.5, b=3, N=200):
    total = 0
    for n in range(N):
        total += -1*a**n * (b**n) * np.pi * np.sin((b**n) * np.pi * x)
    return total

# Second derivative
def d2W_dx2(x, a=0.5, b=3, N=200):
    total = 0
    for n in range(N):
        total += -a**n * (b**(2*n)) * (np.pi**2) * np.cos((b**n) * np.pi * x)
    return total

def gradient_descent(start_x, learning_rate, num_iterations):
    x = start_x
    history = []

    for i in range(num_iterations):
        grad = dW_dx(x)
        x = x - learning_rate * grad
        history.append((x, W(x)))

    return x, W(x), history

start_x = 0.7
learning_rate = 10 ** (-5)
num_iterations = 200

x_opt, w_opt, history = gradient_descent(start_x, learning_rate, num_iterations)

print(f"Final x: {x_opt}")
print(f"Final W(x): {w_opt}")

xs = np.linspace(-2, 2, 5000)
ys = W(xs)

history = np.array(history)

# -----------------------------
# Compute second derivative / sharpness
# across all 200 iterations
# -----------------------------
sharpness_history = []

for i in range(len(history)):
    x = history[i, 0]
    sharpness = d2W_dx2(x)
    sharpness_history.append(sharpness)

sharpness_history = np.array(sharpness_history)

print("\nSharpness across iterations:")
for i, sharpness in enumerate(sharpness_history):
    print(f"Iteration {i + 1}: {sharpness}")

# EoS threshold
eos_threshold = 2 / learning_rate
print(f"\nEoS threshold (2/eta): {eos_threshold}")

plt.plot(xs, ys, label="Weierstrass approximation")
plt.scatter(history[:, 0], history[:, 1], color="red", s=10, label="Gradient descent path")
plt.xlabel("x")
plt.ylabel("W(x)")
plt.legend()
plt.show()
