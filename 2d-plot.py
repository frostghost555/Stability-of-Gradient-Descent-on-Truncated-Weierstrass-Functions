import numpy as np
import matplotlib.pyplot as plt

def W(x, a=0.5, b=3, N=20):
    total = 0
    for n in range(N):
        total += a**n * np.cos((b**n) * np.pi * x)
    return total

def dW_dx(x, a=0.5, b=3, N=20):
    total = 0
    for n in range(N):
        total += -1*a**n * (b**n) * np.pi * np.sin((b**n) * np.pi * x)
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

plt.plot(xs, ys, label="Weierstrass approximation")
plt.scatter(history[:, 0], history[:, 1], color="red", s=10, label="Gradient descent path")
plt.xlabel("x")
plt.ylabel("W(x)")
plt.legend()
plt.show()
