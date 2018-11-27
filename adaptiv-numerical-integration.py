# -*- coding: utf-8 -*-
"""
author: christoph manucredo, chris.manucredo@gmail.com
about:
    this programm implements the numerical adaptiv integration of real
    functions using the Richardson extrapolation as an error estimator.
"""

import numpy as np

f = lambda x: (x**20) * np.cos(x)
h = lambda x: np.sqrt(x+1) * np.cos(x**2)
g = lambda x: (-x**4+x**3-x**2+x)*np.sqrt(x)*np.cos(x)

def quad(a,b,f):
    delta = b-a
    approx = delta*(0.5*f(a+delta*(0.2113248654051871))+0.5*f(a+delta*(0.7886751345948128)))
    return approx
    
def richard(a,b,f):
    I_1 = quad(a,b,f)
    I_2 = quad(a,(a+b)/2,f) + quad((a+b)/2,b,f)
    err = (I_2-I_1)/(15)
    
    return I_2,err

#Usage:
#   -a: Lower Bound of Integral
#   -b: Upper Bound of Integral
#   -f: function to be approximated
#   -tol: desired tolerance of error    
def adaptiv(a,b,f,tol):
    #checking if user inputs something that makes sense
    if a == b:
        return print("Error: Use different bounds for the integral.")
    
    if a > b:
        temp = b
        b = a
        a = temp
    
    #first approximation of the integral
    approx, err = richard(a,b,f)
    value_mat = np.array([[a,b,approx,err]]) #creating the first line of the matrix
    iteration = 1 #a counter to see how many cycles the program runs
    final_approx = 0
    
    if np.abs(err) > tol: #maybe the approximation is good enough on the first try
        a_1 = value_mat[0,0]
        b_1 = value_mat[0,1]
        ab_1 = (a_1 + b_1)/2 #creating a new point in the middle of the initial interval
        approx2, err2 = richard(a_1,ab_1,f) #evaluating with the new bounds
        approx3, err3 = richard(ab_1,b_1,f)
        value_mat[0,:] = [a_1,ab_1,approx2,err2] #rewriting the first line with the first half of the new interval
        newline = np.array([[ab_1,b_1,approx3, err3]]) #creating a new line with the second half of the inverval
        value_mat = np.vstack([value_mat,newline]) #"stacking" the matrix and the new line
    else:
        return print("tolerance too high", value_mat) #this is actually an error message
        
    global_err = np.sum(np.abs(value_mat[:,3])) #adding up all the (absolute) errors
    
    
    while global_err > tol: #this loops as long as the global error is bigger than the user defined tolerance
        highest_err = np.argmax(np.abs(value_mat[:,3])) #looking for the (absolute) biggest error
        a_2 = value_mat[highest_err,0] #reading the values of the interval with the biggest error
        b_2 = value_mat[highest_err,1]
        ab_2 = (a_2+b_2)/2
        approx1, local_error1 = richard(a_2,ab_2,f) #evaluating the new interval
        approx2, local_error2 = richard(ab_2,b_2,f)
        local_error1 = np.abs(local_error1) #making sure that we don't have negative errors
        local_error2 = np.abs(local_error2)
        value_mat[highest_err,:] = [a_2,ab_2, approx1, local_error1] #replacing the line with the biggest error
        newline = np.array([[ab_2,b_2,approx2,local_error2]]) #creating a new line with the second half of the new interval
        value_mat = np.vstack([value_mat,newline]) #stacking the matrix and the new line
        global_err = np.sum(np.abs(value_mat[:,3])) #adding up all the (absolute) errors
        #print("Global Error:", global_err)   #this line is just for debugging reasons 
        iteration += 1 #increasing the counter 
        final_approx = np.sum(value_mat[:,2]) #summing up all the approximations
  
    #Outputs the resulting matrix to a txt for review    
    file = open("output.txt","w")
    file.write("Approximation of an Integral using adaptive integration \n")
    file.write("Lower Bound          Upper Bound      Approximation    Local Error \n")
    file.write(np.array2string(value_mat))
    file.close() 
    
    
    return print("Approximation of the integral:",final_approx,"Iterations:", iteration)
    


