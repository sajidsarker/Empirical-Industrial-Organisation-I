# Import Libraries
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import io
#import statsmodels.api as sm

# Import Dataset
dataset_file = 'airline.csv'
dataset_raw  = open( dataset_file, 'rt' )
dataset_data = np.genfromtxt( dataset_raw, dtype=int, delimiter=',', names=True )

# Dataset Characteristics
N = dataset_data.size

# Generating New ndarray Variables from Dataset
arr_delay    = np.array( dataset_data['ARR_DELAY'] )
dep_delay    = np.array( dataset_data['DEP_DELAY'] )
distance     = np.array( dataset_data['DISTANCE'] )
fe_monday    = np.array( np.empty )
fe_tuesday   = np.array( np.empty )
fe_wednesday = np.array( np.empty )
fe_thursday  = np.array( np.empty )
fe_friday    = np.array( np.empty )
fe_saturday  = np.array( np.empty )
fe_sunday    = np.array( np.empty )

for i in range( N ):
    if ( dataset_data['DAY_OF_WEEK'].item(i) == 1 ):
        fe_monday = np.append( fe_monday, [1] )
        fe_tuesday = np.append( fe_tuesday, [0] )
        fe_wednesday = np.append( fe_wednesday, [0] )
        fe_thursday = np.append( fe_thursday, [0] )
        fe_friday = np.append( fe_friday, [0] )
        fe_saturday = np.append( fe_saturday, [0] )
        fe_sunday = np.append( fe_sunday, [0] )
    if ( dataset_data['DAY_OF_WEEK'].item(i) == 2 ):
        fe_monday = np.append( fe_monday, [0] )
        fe_tuesday = np.append( fe_tuesday, [1] )
        fe_wednesday = np.append( fe_wednesday, [0] )
        fe_thursday = np.append( fe_thursday, [0] )
        fe_friday = np.append( fe_friday, [0] )
        fe_saturday = np.append( fe_saturday, [0] )
        fe_sunday = np.append( fe_sunday, [0] )
    if ( dataset_data['DAY_OF_WEEK'].item(i) == 3 ):
        fe_monday = np.append( fe_monday, [0] )
        fe_tuesday = np.append( fe_tuesday, [0] )
        fe_wednesday = np.append( fe_wednesday, [1] )
        fe_thursday = np.append( fe_thursday, [0] )
        fe_friday = np.append( fe_friday, [0] )
        fe_saturday = np.append( fe_saturday, [0] )
        fe_sunday = np.append( fe_sunday, [0] )
    if ( dataset_data['DAY_OF_WEEK'].item(i) == 4 ):
        fe_monday = np.append( fe_monday, [0] )
        fe_tuesday = np.append( fe_tuesday, [0] )
        fe_wednesday = np.append( fe_wednesday, [0] )
        fe_thursday = np.append( fe_thursday, [1] )
        fe_friday = np.append( fe_friday, [0] )
        fe_saturday = np.append( fe_saturday, [0] )
        fe_sunday = np.append( fe_sunday, [0] )
    if ( dataset_data['DAY_OF_WEEK'].item(i) == 5 ):
        fe_monday = np.append( fe_monday, [0] )
        fe_tuesday = np.append( fe_tuesday, [0] )
        fe_wednesday = np.append( fe_wednesday, [0] )
        fe_thursday = np.append( fe_thursday, [0] )
        fe_friday = np.append( fe_friday, [1] )
        fe_saturday = np.append( fe_saturday, [0] )
        fe_sunday = np.append( fe_sunday, [0] )
    if ( dataset_data['DAY_OF_WEEK'].item(i) == 6 ):
        fe_monday = np.append( fe_monday, [0] )
        fe_tuesday = np.append( fe_tuesday, [0] )
        fe_wednesday = np.append( fe_wednesday, [0] )
        fe_thursday = np.append( fe_thursday, [0] )
        fe_friday = np.append( fe_friday, [0] )
        fe_saturday = np.append( fe_saturday, [1] )
        fe_sunday = np.append( fe_sunday, [0] )
    if ( dataset_data['DAY_OF_WEEK'].item(i) == 7 ):
        fe_monday = np.append( fe_monday, [0] )
        fe_tuesday = np.append( fe_tuesday, [0] )
        fe_wednesday = np.append( fe_wednesday, [0] )
        fe_thursday = np.append( fe_thursday, [0] )
        fe_friday = np.append( fe_friday, [0] )
        fe_saturday = np.append( fe_saturday, [0] )
        fe_sunday = np.append( fe_sunday, [1] )

# Removing Initial Empty Row in ndarray
fe_monday    = np.delete( fe_monday, (0) )
fe_tuesday   = np.delete( fe_tuesday, (0) )
fe_wednesday = np.delete( fe_wednesday, (0) )
fe_thursday  = np.delete( fe_thursday, (0) )
fe_friday    = np.delete( fe_friday, (0) )
fe_saturday  = np.delete( fe_saturday, (0) )
fe_sunday    = np.delete( fe_sunday, (0) )


# Question 1:
print('Question 1:')
print('')

print( "N = " + str( N ) )
print('')

# Define in-line SSE function for minimisation
# Fixed Effects for Sunday excluded in model specification
x0 = np.ones( shape=(arr_delay.size, ) )
f_sse = lambda b: np.sum( np.square( arr_delay - b[0] * x0 - b[1] * distance - b[2] * dep_delay - b[3] * fe_monday - b[4] * fe_tuesday - b[5] * fe_wednesday - b[6] * fe_thursday - b[7] * fe_friday - b[8] * fe_saturday ) )

# Minimise defined OLS function
# Using FMinSearch
print('Minimisation using FMin')
sse = sp.optimize.fmin( f_sse, [0, 0, 0, 0, 0, 0, 0, 0, 0] )
print( sse )
for i in range(9):
    print( "Beta " + str(i) + ": " + str(sse[i]) )
print('')

# Using Basin-Hopping
print('Minimisation using Basin-Hopping')
sse = sp.optimize.basinhopping( f_sse, [0, 0, 0, 0, 0, 0, 0, 0, 0], 4 )
print( sse.x )
for i in range(9):
    print( "Beta " + str(i) + ": " + str(sse.x[i]) )
print('')
print('')

# Comparison to OLS regression
#regressors = np.concatenate( ( x0, np.array( distance, dtype=float ), np.array( dep_delay, dtype=float ), np.array( fe_monday, dtype=float ), np.array( fe_tuesday, dtype=float ), np.array( fe_wednesday, dtype=float ), np.array( fe_thursday, dtype=float ), np.array( fe_friday, dtype=float ), np.array( fe_saturday, dtype=float ) ), axis=0 )
#regressors = ( np.reshape( regressors, (9, N) ) ).T
#regression = sm.OLS( exog=arr_delay, endog=regressors, hasconst=True )
#reg_fit    = regression.fit()
#print( reg_fit.summary() )


# Question 2:
print('Question 2:')
print('')

# Generate Binary Variable for Flights Arriving Later than 15 minutes
arr_late = np.array( np.empty, dtype=bool )
for i in range(N):
    if ( arr_delay[i] > 15 ):
        arr_late = np.append( arr_late, [1] )
    else:
        arr_late = np.append( arr_late, [0] )
arr_late = np.delete( arr_late, (0) )

#Define in-line function for minimisation
f_mle = lambda b: np.sum( -arr_late * ( b[0] * x0 + b[1] * distance + b[2] * dep_delay ) + np.log( 1 + np.exp( b[0] * x0 + b[1] * distance + b[2] * dep_delay ) ) )

# Minimise defined MLE function
# Using FMinSearch
print('Minimisation using FMin')
mle = sp.optimize.fmin( f_mle, [0, 0, 0] )
print( mle )
for i in range(3):
    print( "Beta " + str(i) + ": " + str(mle[i]) )
print('')

# Using Minimise
print('Minimisation using Minimise')
mle = sp.optimize.minimize( f_mle, [0, 0, 0] )
print( mle )
for i in range(3):
    print( "Beta " + str(i) + ": " + str(mle.x[i]) )
print('')
print('')


# Question 3:
print('Question 3:')
print('')

# Import Dataset
dataset_file = 'IV.mat'
dataset_raw  = sp.io.loadmat( dataset_file )

# Generating New ndarray Variables from Dataset
sp.io.whosmat( dataset_file )
Y  = np.array( dataset_raw['Y'] )
X  = np.array( dataset_raw['X'] )
X0 = X[:,0]
X1 = X[:,1]
X2 = X[:,2]
X0 = np.reshape( X0, (-1, 1) )
X1 = np.reshape( X1, (-1, 1) )
X2 = np.reshape( X2, (-1, 1) )
Z  = np.array( dataset_raw['Z'] )
Z0 = Z[:,0]
Z1 = Z[:,1]
Z2 = Z[:,2]
Z3 = Z[:,3]
Z0 = np.reshape( Z0, (-1, 1) )
Z1 = np.reshape( Z1, (-1, 1) )
Z2 = np.reshape( Z2, (-1, 1) )
Z3 = np.reshape( Z3, (-1, 1) )
I  = np.identity( 4 )

# Dataset Characteristics
N = Y.size
print( "N = " + str( N ) )
print('')

#Define first stage function to be minimised
def f_gmm1(b):
    bX  = np.matmul( b, X.T )
    bX  = bX.T
    e   = np.subtract( Y, bX )
    e   = np.diag( e )
    gw  = np.matmul( Z.T, e )
    return np.matmul( gw.T, gw )

gmm1 = sp.optimize.minimize( f_gmm1, [0, 0, 0] )
print( gmm1 )
for i in range(3):
    print( "Beta " + str(i) + ": " + str(gmm1.x[i]) )
print('')

e = np.matmul( gmm1.x, X.T )
e = e.T
e = np.subtract( Y, e )
e = np.diag( e )

# Constructing Weight Matrix
W = np.zeros( (4, 4) )
for i in range(N):
    w = e[i] * Z[i,:]
    w = np.reshape( w, (4, 1) )
    W = W + np.dot( w, w.T )
W = np.linalg.inv( W )

# Computing Standard Errors
Q = np.matmul( Z.T, X )
C = np.dot( np.dot( Q.T, W ), Q )
gmm1_variance = np.linalg.inv( C )

print( 'GMM Stage 1 Variance' )
print( gmm1_variance )
print('')

gmm1_stderror = gmm1_variance
for j in range(3):
    for i in range(3):
        gmm1_stderror[i, j] = np.sqrt( gmm1_variance[i, j] )

print( 'GMM Stage 1 Standard Errors' )
print( gmm1_stderror )
print('')

#Define second stage function to be minimised
def f_gmm2(b):
    bX  = np.matmul( b, X.T )
    bX  = bX.T
    e   = np.subtract( Y, bX )
    e   = np.diag( e )
    gw  = np.matmul( Z.T, e )
    out = np.matmul( gw.T, W )
    out = np.matmul( out, gw )
    return out

gmm2 = sp.optimize.minimize( f_gmm2, [0, 0, 0] )
print( gmm2 )
for i in range(3):
    print( "Beta " + str(i) + ": " + str(gmm2.x[i]) )
print('')

e = np.matmul( gmm2.x, X.T )
e = e.T
e = np.subtract( Y, e )
e = np.diag( e )

# Constructing Weight Matrix
W = np.zeros( (4, 4) )
for i in range(N):
    w = e[i] * Z[i,:]
    w = np.reshape( w, (4, 1) )
    W = W + np.dot( w, w.T )
W = np.linalg.inv( W )

# Computing Standard Errors
Q = np.matmul( Z.T, X )
C = np.dot( np.dot( Q.T, W ), Q )
gmm2_variance = np.linalg.inv( C )

print( 'GMM Stage 2 Variance' )
print( gmm2_variance )
print('')

gmm2_stderror = gmm2_variance
for j in range(3):
    for i in range(3):
        gmm2_stderror[i, j] = np.sqrt( gmm2_variance[i, j] )

print( 'GMM Stage 2 Standard Errors' )
print( gmm2_stderror )
print('')

# Comparison of Standard Errors
print( 'GMM Variance Difference' )
print( gmm2_variance - gmm1_variance )
print('')

print( 'GMM Standard Errors Difference' )
print( gmm2_stderror - gmm1_stderror )
print('')

# EOF
