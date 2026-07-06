import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Deeply coupled Weierstrass surface
# -----------------------------
def F(x, y, a=0.5, b=3, N=15):
    total = 0
    for n in range(N):
        total += (
            a**n
            * np.cos((b**n) * np.pi * x)
            * np.cos((b**n) * np.pi * y)
        )
    return total


# -----------------------------
# Gradient of F
# -----------------------------
def grad_F(x, y, a=0.5, b=3, N=15):
    gx = 0
    gy = 0

    for n in range(N):
        coeff = a**n * (b**n) * np.pi

        gx += (
            -coeff
            * np.sin((b**n) * np.pi * x)
            * np.cos((b**n) * np.pi * y)
        )

        gy += (
            -coeff
            * np.cos((b**n) * np.pi * x)
            * np.sin((b**n) * np.pi * y)
        )

    return gx, gy


# -----------------------------
# Gradient Descent
# -----------------------------
def gradient_descent(start_x, start_y, learning_rate, num_iterations):
    x = start_x
    y = start_y

    history = []

    for i in range(num_iterations):

        gx, gy = grad_F(x, y)

        x -= learning_rate * gx
        y -= learning_rate * gy

        history.append((x, y, F(x, y)))

    return x, y, F(x, y), history


# -----------------------------
# Parameters
# -----------------------------
start_x = 0.7
start_y = -0.5

learning_rate = 1e-4
num_iterations = 200

x_opt, y_opt, f_opt, history = gradient_descent(
    start_x,
    start_y,
    learning_rate,
    num_iterations,
)

print(f"Final x = {x_opt}")
print(f"Final y = {y_opt}")
print(f"Final F(x,y) = {f_opt}")

# -----------------------------
# Plot the surface
# -----------------------------
xs = np.linspace(-2, 2, 300)
ys = np.linspace(-2, 2, 300)

X, Y = np.meshgrid(xs, ys)
Z = F(X, Y)

history = np.array(history)

best_idx = np.argmin(history[:, 2])

print("\nBest point reached:")
print("iteration =", best_idx)
print("x =", history[best_idx, 0])
print("y =", history[best_idx, 1])
print("F(x,y) =", history[best_idx, 2])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Surface
ax.plot_surface(
    X,
    Y,
    Z,
    cmap="viridis",
    alpha=0.8,
    linewidth=0
)

# Gradient descent trajectory
ax.plot(
    history[:, 0],
    history[:, 1],
    history[:, 2],
    color="red",
    linewidth=2,
    label="Gradient Descent"
)

ax.scatter(
    history[:, 0],
    history[:, 1],
    history[:, 2],
    color="red",
    s=10
)

# Final point
ax.scatter(
    history[-1, 0],
    history[-1, 1],
    history[-1, 2],
    color="black",
    s=80,
    label="Final Point"
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("F(x,y)")
ax.set_title("Gradient Descent on Coupled Weierstrass Surface")

plt.show()
