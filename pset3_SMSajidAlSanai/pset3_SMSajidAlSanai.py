# Import Libraries
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import io
import matplotlib.pyplot as plt
from matplotlib import colors

# Question 1:
print('Question 1:')
print('')

# Estimation of Theta3
def MaximumLogLikelihoodTheta3( probability, setting ):
    out = 0
    outcome  = np.array( setting[0:3] )
    binomial = np.array( [ np.sum( outcome ), setting[3] ] )
    binomial_coefficient = sp.special.binom( binomial[0], binomial[1] )
    for i in range( np.size( probability ) ):
        out += np.log( probability[i] ) * outcome[i]
    out += np.log( 1 - ( probability[0] + probability[1] ) ) * outcome[2]
    out += np.log( binomial_coefficient )
    return out

def MaximumLikelihoodTheta3( probability, setting ):
    outcome  = np.array( setting[0:3] )
    binomial = np.array( [ np.sum( outcome ), setting[3] ] )
    binomial_coefficient = sp.special.binom( binomial[0], binomial[1] )
    out = binomial_coefficient
    for i in range( np.size( probability ) ):
        out *= probability[i] ** outcome[i]
    out *= ( 1 - ( probability[0] + probability[1] ) ) ** outcome[2]
    out *= binomial_coefficient
    return out

# Flow Payoff
def FlowPayoff( parameter, state, x ):
    out = -parameter[0] * state - parameter[1] * ( 1 - state ) * x
    return out

# Inner Loop
def InnerLoop( parameter, setting ):
    n              = setting[0]
    tolerance      = setting[1]
    max_iterations = setting[2]
    sup_difference = 1
    expected_value = np.zeros( (n, 2) )
    curr_iteration = 1

    while sup_difference > tolerance and curr_iteration < max_iterations:
        x = np.arange( n )
        t_expected_value = np.zeros( (n, 2) )
        # Do not replace
        t_expected_value[:, 0] = np.log( np.exp( FlowPayoff( parameter, 0, x ) + parameter[2] * expected_value[:, 0] ) + np.exp( FlowPayoff( parameter, 1, x ) + parameter[2] * expected_value[:, 1] ) ).T
        # Replace
        t_expected_value[:, 1] = np.log( np.exp( FlowPayoff( parameter, 0, x ) + parameter[2] * expected_value[0, 0] ) + np.exp( FlowPayoff( parameter, 1, x ) + parameter[2] * expected_value[0, 1] * np.ones( n, ) ) ).T
        # Expected Value
        t_expected_value = np.matmul( transition_probability, t_expected_value )
        t_expected_value[n-2:, :] = np.array( [ 0.5 * (t_expected_value[n-3,0] + t_expected_value[n-1,0]), 0.5 * (t_expected_value[n-3,1] + t_expected_value[n-1,1]) ] )
        sup_difference = np.max( abs( t_expected_value - expected_value ) )
        expected_value = t_expected_value
        # Increment Tau
        curr_iteration += 1
    out = expected_value
    print( 'Iteration [' + str(curr_iteration) + ']:' )
    print( out )
    # Returns (Expected Value)
    return out

# Outer Loop
def OuterLoop( parameter, setting ):
    max_iterations = int( setting[0] )
    tolerance      = setting[1]
    n              = int( setting[2] )
    # [ Replacement Cost, Maintenance Cost ]: Initial Guess Vector (Theta)
    # [ i_t, x_t ]: Vector of State Variables at Time t
    out = InnerLoop( parameter, [n, tolerance, max_iterations] )
    '''
    for i in range( n ):
        print( 'Iteration [' + str(i) + ']' )
        out[i, :] = InnerLoop( parameter, [i, tolerance, max_iterations, n] )
        #out[t, :] = sp.optimize.minimize( InnerLoop, [1, 1], [state_i[t], state_x[t], tolerance, max_iterations] )
    '''
    # Returns (Expected Values)
    # Returns (Theta)
    return out

# Choice Probability
def ChoiceProbability( parameter, x ):
    #out = 1 / ( 1 + np.exp( parameter[0] + parameter[1] * x - parameter[2] * expected_values[:, 0] - parameter[1] * x + parameter[2] * expected_values[1, 1] ) )
    #out = 1 / (1 + np.exp(-parameter[0] + parameter[2] * expected_values[0, 1] + parameter[1] * x - parameter[2] * expected_values[:, 0]))
    if parameter[6] == 0:
        out1 =  np.exp( -parameter[1] * x + parameter[2] * expected_values[:, 0] )
        out2 =  np.exp( -parameter[0] + parameter[2] * expected_values[1, 1] )
        out2 += out1
        out = out1 / out2
    if parameter[6] == 1:
        out1 =  np.exp( -parameter[0] + parameter[2] * expected_values[1, 1] )
        out2 =  np.exp( -parameter[1] * x + parameter[2] * expected_values[:, 0] )
        out2 += out1
        out = out1 / out2
    return out

# Main Program Entry Point
n = 11

#state_i = np.zeros( (S,) ) # state_i[:]: i_t
#state_x = np.zeros( (S,) ) # state_x[:]: x_t | i_t = 0
#for s in range( S ):
#    out = np.random.multinomial( 1, [0.3, 0.5, 0.2] )
#    state_i[s] = 0
#    state_x[s] = np.matmul( np.array( [0, 1, 2] ).T, out )

cost_maintenance = 0.05
cost_replacement = 10
beta   = 0.99
theta3 = np.array( [0.3, 0.5, 0.2] )

print( 'cost_replacement=' + str(cost_replacement) )
print( 'cost_maintenance=' + str(cost_maintenance) )
print( 'beta=' + str(beta) )
print( 'theta3=' + str(theta3) )
print('')

# Construct Transition Probability Matrix
transition_probability = np.zeros( (n, n) )
p = np.array( (theta3[0], theta3[1], theta3[2], 0, 0, 0, 0, 0, 0, 0, 0) )
transition_probability = np.tile( p, (1, n) )
transition_probability = np.reshape( transition_probability, (n, n) )
for i in range(n):
    transition_probability[i, :] = np.roll( transition_probability[i, :], i)
transition_probability[ 9, 0], transition_probability[10, 0:2] = 0, 0
transition_probability[ 9, n-1] = 1 - transition_probability[9, n-2]
transition_probability[10, 10] = 1

print( 'Transition Probability Matrix (n x n)' )
print( transition_probability )
print('')

# Run Outer Loop
expected_values = OuterLoop( np.array( [cost_replacement, cost_maintenance, beta, theta3[0], theta3[1], theta3[2]] ), np.array( [1000, 0.0001, n] ) )
print('')

# Plot Expected Values
plt.plot( expected_values[:, 0] )
plt.plot( expected_values[:, 1] )
plt.title('Expected Values')
plt.xlabel('Mileage Bin')
plt.ylabel('US Dollars')
plt.legend(('Do Not Replace', 'Replace'))
plt.show()

# Calculate Choice Probabilities
choice_probability0 = ChoiceProbability( np.array( [cost_replacement, cost_maintenance, beta, theta3[0], theta3[1], theta3[2], 0] ), np.arange(n) )
choice_probability1 = ChoiceProbability( np.array( [cost_replacement, cost_maintenance, beta, theta3[0], theta3[1], theta3[2], 1] ), np.arange(n) )

print( 'Choice Probability (No Replacement)' )
print( choice_probability0)
plt.plot( choice_probability0 )
plt.title( 'Choice Probability (No Replacement)' )
plt.xlabel('Mileage Bin')
plt.ylabel('Probability')
plt.show()
print('')

print( 'Choice Probability (Replacement)' )
print( choice_probability1)
plt.plot( choice_probability1 )
plt.title( 'Choice Probability (Replacement)' )
plt.xlabel('Mileage Bin')
plt.ylabel('Probability')
plt.show()
print('')

# Simulation of Dataset
N = 100
S = 1000

# Set Seed Value
np.random.seed(23061994)

state_x = np.zeros( (S, N) )

for k in range(N):
    #state_x[0, k] = int(np.random.uniform(0, 10))
    state_x[0, k] = np.random.randint(0, 7)

for j in range(N):
    for l in range(S-1):
        uniform_draw = np.random.uniform(0, 1)
        #print( uniform_draw )
        num1 = 0
        num2 = 0
        depen_x = int(state_x[l, j])
        #state_x[ l+1, j ] = 0
        for k in range(n):
            num1 = np.sum( transition_probability[ depen_x, 0:max(0, k-1) ] )
            num2 = np.sum( transition_probability[ depen_x, 0:k ] )
            #print( str(num1) + '<' + str(uniform_draw) + '<=' + str(num2) )
            if uniform_draw > num1 and uniform_draw <= num2:
                state_x[l+1, j] = int(k)
        #print( state_x[l+1, j] )

print( 'Summary Statistics of State Variable x[t] across [100] Buses' )
print( 'Mean: ' + str( np.mean( state_x ) ) )
print( 'Std:  ' + str( np.std( state_x ) ) )
print( 'Max:  ' + str( np.max( state_x ) ) )
print( 'Min:  ' + str( np.min( state_x ) ) )
print('')

state_epsilon = np.zeros( (S, 2) )

for t in range( S ):
    state_epsilon[t, 0] = np.random.gumbel( 0, 1 ) - 0.57721
    state_epsilon[t, 1] = np.random.gumbel( 0, 1 ) - 0.57721

for i in range(2):
    print( '[Epsilon(' + str(i) + ')]' )
    print( 'Mean: ' + str( np.mean(state_epsilon[:, i]) ) )
    print( 'Max: '  + str( np.max(state_epsilon[:, i]) ) )
    print( 'Min: '  + str( np.min(state_epsilon[:, i]) ) )
    print('')

state_i = np.zeros( (S, N) )


# Question 2
print('Question 2:')
print('')

# Estimation of Theta3
theta3 = np.zeros( (3, 3) )
for k in range(3):
    out = sp.optimize.minimize( MaximumLikelihoodTheta3, [0, 0], [0, 1, 2, k] )
    theta3[k, 0:2] = out.x
    theta3[k,   2] = 1 - np.sum( theta3[k, 0:2] )
print( 'theta3=' + str(theta3) )
print('')

cost_maintenance = 0.02
cost_replacement = 20
beta   = 0.99
theta3 = np.array( [0.3, 0.5, 0.2] )

print( 'cost_replacement=' + str(cost_replacement) )
print( 'cost_maintenance=' + str(cost_maintenance) )
print( 'beta=' + str(beta) )
print( 'theta3=' + str(theta3) )
print('')

# Construct Transition Probability Matrix
transition_probability = np.zeros( (n, n) )
p = np.array( (theta3[0], theta3[1], theta3[2], 0, 0, 0, 0, 0, 0, 0, 0) )
transition_probability = np.tile( p, (1, n) )
transition_probability = np.reshape( transition_probability, (n, n) )
for i in range(n):
    transition_probability[i, :] = np.roll( transition_probability[i, :], i)
transition_probability[ 9, 0], transition_probability[10, 0:2] = 0, 0
transition_probability[ 9, n-1] = 1 - transition_probability[9, n-2]
transition_probability[10, 10] = 1

print( 'Transition Probability Matrix (n x n)' )
print( transition_probability )
print('')

# Run Outer Loop
expected_values = OuterLoop( np.array( [cost_replacement, cost_maintenance, beta, theta3[0], theta3[1], theta3[2]] ), np.array( [1000, 0.0001, n] ) )
print('')

# Plot Expected Values
plt.plot( expected_values[:, 0] )
plt.plot( expected_values[:, 1] )
plt.title('Expected Values')
plt.xlabel('Mileage Bin')
plt.ylabel('US Dollars')
plt.legend(('Do Not Replace', 'Replace'))
plt.show()

print('')
print('***')
print('[EOF] Output Terminates')
print('***')
# EOF
