import matplotlib.pyplot as plt
import numpy as np
import math

# features:- size of the property
x_train = np.array([400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300,
    1400, 1500, 1600, 1700, 1800, 1900, 2000, 2200, 2400,
    2600, 2800, 3000, 3200, 3500, 3800, 4200, 4600, 5000])

# target_value:- cost of the property in lakhs of $
y_train = np.array([18, 22, 26, 30, 35, 40, 45, 50, 55, 60,
    65, 70, 75, 80, 86, 92, 98, 110, 122,
    135, 148, 160, 175, 195, 215, 240, 265, 290])

x_scaled = x_train/max(x_train)
y_scaled = y_train/max(y_train)



def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    total_cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i])**2
        total_cost = cost/(2*m)

    return total_cost;

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw,dj_db;

def gradient_descent(w_in,b_in,iteration,x,y,alpha,compute_cost,compute_gradient):
    """Computing the gradent descent to fix the w,b in data set so well
    w_in = w
    b_in = b
    iteration = no of time the algo will run to compute cost and run well 
    x = x_train data set 
    y = y_train data set
    alpha = learning set 
    compute_cost = cost_function  = (1/2m ∑ m ,i =1 (ƒ(wb)x[i]) - y[i])**2 )
    gradient = gradient function ∂J(w,b)/∂w = (1/m ∑ m ,i=1 (ƒ(wb) X[i]) - y[i])* X[i])
                                 ∂J(w,b)/∂b = (1/m ∑ m ,i=1 (ƒ(wb) X[i]) - y[i]))

    Returning values = 
    w_scalar = w value after running gradient descent 
    b_scalar = b walue after running gradient descent 
    J_history (list) = history of cost function
    p_history (list) = history of parameter function"""

    j_history = []
    p_history = []
    w = w_in
    b = b_in

    for i in range(iteration):
        dj_dw, dj_db = compute_gradient(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            j_history.append(compute_cost(x,y,w,b))
            p_history.append([w,b])
        if i % math.ceil(iteration/10) == 0:
            print(f"Iteration {i}: "
                  f"Cost = {j_history[-1]:.4f}, "
                  f"dj_dw = {dj_dw:.4f}, dj_db = {dj_db:.4f},"
                  f"w = {w:.4f}, b = {b:.4f}"
                 )


    
    return j_history,p_history,w,b;

w_init = 0
b_init = 0
iteration = 100000
tm_aphla = 1.0e-2
j_hist, p_hist,w_final, b_final,  = gradient_descent(w_init,b_init,iteration,x_scaled,y_scaled,tm_aphla,     compute_cost,compute_gradient) 
print(f"w and b found on {w_final:8.4f},{b_final:8.4f}")


def predict(x,w,b):
    return w * x + b

y_hat = predict(w_final,x_scaled,b_final)

# raw data 
plt.figure(figsize=(5,8))
plt.scatter(x_scaled,y_scaled, label = "actual data")
plt.xlabel("Property Size (sq ft")
plt.ylabel("Property Cost (Lakhs)")
plt.title("Raw data set")
plt.legend()
plt.show()
plt.close()

#predicted values
plt.figure(figsize=(5,8))
plt.scatter(x_scaled,y_scaled, label = "actual data")
plt.plot(x_scaled,y_hat, color = "red", label = "Predicted Line" )
plt.xlabel("Property Size (sq ft)")
plt.ylabel("Property Cost (Lakhs)")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
plt.close()

#gradient descent progress
plt.figure(figsize=(5,8))
plt.plot(j_hist)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent: Cost vs Iterations")
plt.show()
plt.close()

plt.figure(figsize=(5,8))
plt.scatter(y_scaled,y_hat)
plt.xlabel("target value")
plt.ylabel("predict value")
plt.title("Actual vs predicted")
plt.show()
plt.close()
