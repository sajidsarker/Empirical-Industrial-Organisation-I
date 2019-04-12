# Import Libraries
import math
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import io
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import colors

# Question 1:
print('Question 1:')
print('')

# Import Dataset
dataset_file = 'ascending_data.dat'
dataset_raw  = open( dataset_file, 'rt' )
dataset_data = np.genfromtxt( dataset_raw, dtype=(float, float), delimiter=None, names='num_bidders, price_paid' )

# Dataset Characteristics
T = dataset_data.size

# Generating New ndarray Variables from Dataset
num_bidders = np.reshape( dataset_data['num_bidders'], (T, 1) )
price_paid  = np.reshape( dataset_data['price_paid'],  (T, 1) )

# Dataset Characteristics
Tn_value, Tn_num = np.unique( num_bidders, return_counts=True )

print( 'T  = ' + str( T ) )
print( 'Tn = ' + str( Tn_value ) )
print( '     ' + str( Tn_num ) )
print('')

# GPV Non-Parametric Estimation Method
def ObtainPseudoPrivateValue( bids, i, cdf, pdf ):
    out = np.zeros( (Tn_num[0],) )
    for bi in range( Tn_num[i] ):
        Gb = ObtainKernelCDF( cdf, bi / Tn_num[i] )
        gb = ObtainKernelpdf( bids[Tn_num[i] * i + bi] )
        out[bi] = bids[Tn_num[i] * i + bi] - ( 1 / ( Tn_value[i] - 1 ) ) * ( Gb / gb )
    return out

def ObtainKernelpdf( bid ):
    density = sm.nonparametric.KDEUnivariate( price_paid )
    density.fit()
    return density.evaluate( bid )

def ObtainKernelCDF( cdf, bid ):
    i = int( np.round( bid * cdf.shape[0] ) )
    return cdf[i]

def EstimateKernelpdf( arr ):
    density = sm.nonparametric.KDEUnivariate( arr )
    density.fit()
    t_cdf = density.cdf
    out   = np.zeros( t_cdf.shape )
    out[0] = t_cdf[0]
    for i in range( (density.cdf).shape[0] - 1 ):
        out[i + 1] = t_cdf[i + 1] - t_cdf[i]
    return out

def EstimateKernelCDF( arr ):
    density = sm.nonparametric.KDEUnivariate( arr )
    density.fit()
    return density.cdf

# Conduct Kernel Density Distribution
bids_pdf = np.zeros( (512, Tn_value.size) )
bids_cdf = np.zeros( (512, Tn_value.size) )
for i in range( Tn_value.size ):
    bids_pdf[:, i] = EstimateKernelpdf( price_paid[Tn_num[i] * i : Tn_num[i] * (i + 1)] )
    bids_cdf[:, i] = EstimateKernelCDF( price_paid[Tn_num[i] * i : Tn_num[i] * (i + 1)] )

# Draw Graph of Cumulative Distribution
print( 'Outputting pdf of Bids ...' )
print( bids_pdf.shape )
print( bids_pdf )
print('')

print( 'Generating Plot of Distribution for pdf of Bids ...' )
for i in range( Tn_value.size ):
    plt.plot( bids_pdf[:, i], label='n=' + str(3+i) )
print( 'Graph [' + str(i+1) + ']' )
plt.title('Non-Parametric Probability Density Function for Bids (' + str(3+i) + ' bidders)')
plt.xlabel('v')
plt.ylabel('Pr(Valuation=v)')
plt.legend()
plt.show()
print('')

# Draw Graph of Cumulative Distribution
print( 'Outputting CDF of Bids ...' )
print( bids_cdf.shape )
print( bids_cdf )
print('')

print( 'Generating Plot of Distribution for CDF of Bids ...' )
for i in range( Tn_value.size ):
    plt.plot( bids_cdf[:, i], label='n=' + str(3+i) )
print( 'Graph [' + str(i+1) + ']' )
plt.title('Non-Parametric Cumulative Distribution Function for Bids (' + str(3+i) + ' bidders)')
plt.xlabel('v')
plt.ylabel('Pr(Valuation<=v)')
plt.legend()
plt.show()
print('')

# Estimate Pseudo Private Values
pseudo_private_values = np.zeros( (Tn_num[0], Tn_value.size) )
for i in range( Tn_value.size ):
    pseudo_private_values[:, i] = ObtainPseudoPrivateValue( price_paid, i, bids_pdf[:, i], bids_cdf[:, i] )
print( 'Outputting Pseudo Private Values ...' )
print( pseudo_private_values )
print('')

# Draw Graph of Cumulative Distribution
print( 'Generating Histogram of Distribution for Pseudo Private Values ...' )
num_bins = 25
for i in range( Tn_value.size ):
    print( 'Graph [' + str(i+1) + ']' )
    N, bins, patches = plt.hist(pseudo_private_values[:, i], bins=num_bins)
    fracs = N / N.max()
    norm  = colors.Normalize( fracs.min(), fracs.max() )
    for ifrac, ipatch in zip( fracs, patches ):
        color = plt.cm.viridis( norm( ifrac ) )
        ipatch.set_facecolor( color )
    plt.title('Distribution of Pseudo Private Values (' + str(3+i) + ' bidders)')
    plt.xlabel('Estimated Pseudo Private Values')
    plt.ylabel('Frequency')
    plt.show()
print('')
print('')


# Question 2:
print('Question 2:')
print('')

# Haile & Tamer Estimation Method
def EstimateG( n, v, delta ):
    out   = 0
    t_sum = 0
    condition_one = False
    condition_two = False
    for t in range( Tn_num[0] ):
        t_index = (np.where(Tn_value == n))[0] * Tn_num[0] + t
        if num_bidders[ t_index ] == n:
            condition_one = True
        if price_paid[ t_index ] + delta <= v:
            condition_two = True
        if condition_one == True and condition_two == True:
            t_sum += 1
        condition_one = False
        condition_two = False
    out = ( 1 / Tn_num[ (np.where(Tn_value == n))[0] ] ) * t_sum
    return out

def MonotoneOperation( cdf, i ):
    out         = np.zeros( (cdf.shape[0], cdf.shape[1]) )
    coefficient = np.zeros( (Tn_value.size,) )
    for k in range( Tn_value.size ):
        for j in range( Tn_num[0] ):
            t_polynom = np.zeros( (int(Tn_value[k]) + 1,) )
            if Tn_value[k] == 3:
                t_polynom = [-1/3, 1/2, 0, ( math.factorial( 3 - i ) * math.factorial( i - 1 ) ) / math.factorial( 3 )]
            if Tn_value[k] == 4:
                t_polynom = [1/4, -2/3, 1/2, 0, ( math.factorial( 4 - i ) * math.factorial( i - 1 ) ) / math.factorial( 4 )]
            if Tn_value[k] == 5:
                t_polynom = [-1/5, 3/4, -1, 1/2, 0, ( math.factorial( 5 - i ) * math.factorial( i - 1 ) ) / math.factorial( 5 )]
            t_polynom[ int(Tn_value[k]) ] *= -cdf[j, k]
            t_root = np.roots( t_polynom )
            t_root = np.unique( np.real( t_root[ (t_root >= 0) & (t_root <= 1) ] ) )
            out[j, k] = np.max( t_root )
    return out

def WeightedAverage( y, rho ):
    t_result_vector1 = np.zeros( (Tn_num[0], Tn_value.size) )
    t_result_vector2 = np.zeros( (Tn_num[0], Tn_value.size) )
    t_result_vector3 = np.zeros( (Tn_num[0], Tn_value.size) )
    t_denominator    = np.zeros( (Tn_num[0],) )
    out              = np.zeros( (Tn_num[0],) )
    for k in range( Tn_num[0] ):
        for j in range( Tn_value[0].size ):
            t_result_vector1[k, j] = np.exp( y[k, j] * rho )
            t_result_vector2[k, j] = np.exp( y[k, j] * rho ) * y[k, j]
        t_denominator[k] = np.sum( t_result_vector1[k, :] )
        t_result_vector3[k, j] = t_result_vector2[k, j] / t_denominator[k]
        out[k] = np.sum( t_result_vector3[k, :] )
    return out

# Estimate our distributions over differing values v for each n
DELTA = 1.0
CDFU  = np.zeros( (Tn_num[0], Tn_value.size) )
CDFL  = np.zeros( (Tn_num[0], Tn_value.size) )

for i in range( Tn_value.size ):
    for v in range( Tn_num[0] ):
        CDFU[v, i] = EstimateG( Tn_value[i], v, 0 )
        CDFL[v, i] = EstimateG( Tn_value[i], v, DELTA )

print( '[G^_i=2:n=3] ' + str( CDFU[:, 0].shape ) )
print( '[G^_i=2:n=4] ' + str( CDFU[:, 1].shape ) )
print( '[G^_i=2:n=5] ' + str( CDFU[:, 2].shape ) )
print('')

print( '[G^_i=2:n=3,4,5]' )
print(CDFU)
print('')

print( '[G^_i=3:n=3:del=1] ' + str( CDFL[:, 0].shape ) )
print( '[G^_i=4:n=4:del=1] ' + str( CDFL[:, 1].shape ) )
print( '[G^_i=5:n=5:del=1] ' + str( CDFL[:, 2].shape ) )
print('')

print( '[G^_i=n:n=3,4,5:del=1]' )
print(CDFL)
print('')

print('Calculating Estimated Distributions forming Upper and Lower Bounds ...')
FU = WeightedAverage( MonotoneOperation( CDFU, 2 ), -700 )
FL = WeightedAverage( CDFL, 700 )

print( 'Generating Plot of Bounds on Distribution for CDF of Bids with Smoothing ...' )
plt.plot(FU)   # Blue   # Issue lay here previously
plt.plot(FL)   # Orange
plt.title('Haile & Tamer Distribution Bounds for CDF of Bids with Smoothing')
plt.xlabel('v')
plt.ylabel('Pr(Valuation<=v)')
plt.show()

print('')
print('')


# Question 3:
print('Question 3:')
print('')

# Import Dataset
dataset_file = 'fpa.dat'
dataset_raw  = open( dataset_file, 'rt' )
dataset_data = np.genfromtxt( dataset_raw, dtype=(float, float), delimiter=None, names='bids1, bids2, bids3, bids4' )

# Dataset Characteristics
num_auctions = dataset_data.shape[0]
num_bidders  = 4

print( 'No. of Auctions: ' + str( num_auctions ) )
print( 'No. of Bidders:  ' + str( num_bidders  ) )
print('')

# Generating New ndarray Variables from Dataset
bids = np.zeros( shape=(num_auctions, num_bidders) )
bids[:, 0] = np.reshape( dataset_data['bids1'], (num_auctions,) )
bids[:, 1] = np.reshape( dataset_data['bids2'], (num_auctions,) )
bids[:, 2] = np.reshape( dataset_data['bids3'], (num_auctions,) )
bids[:, 3] = np.reshape( dataset_data['bids4'], (num_auctions,) )

def EstimateG( v ):
    out   = 0
    t_sum = 0
    for l in range( num_auctions ):
        for p in range( num_bidders ):
            if bids[l, p] <= v:
                t_sum += 1
    out = ( 1 / ( num_auctions * num_bidders ) ) * t_sum
    return out

def Estimateg( v ):
    out   = 0
    t_sum = 0
    for l in range( num_auctions ):
        for p in range( num_bidders ):
            #h_g
            bandwidth = 2.978 * 1.06 * np.std( bids[:, p] )
            normalise = ( v - bids[l, p] ) / bandwidth
            t_sum += ( 1 / ( num_auctions * num_bidders * bandwidth ) ) * KernelFunction( normalise )
    out = t_sum
    return out

def ObtainFPSBPrivateValue():
    out = np.zeros( shape=(num_auctions, num_bidders) )
    for j in range( num_auctions ):
        for i in range( num_bidders ):
            out[j, i] = bids[j, i] - ( 1 / (num_bidders - 1) ) * ( EstimateG( bids[j, i] ) / Estimateg( bids[j, i] ) )
    return out

def ObtainFPSBf( v ):
    out   = 0
    t_sum = 0
    for j in range( num_auctions ):
        for i in range( num_bidders ):
            R = 2
            #h_f
            bandwidth = 2.978 * 1.06 * np.std( fpsb_private_values[:, i] ) * ( np.log( num_auctions ) / num_auctions )**( 1 / ( 2 * R + 2 * num_bidders - 2 ) )
            normalise = (v - fpsb_private_values[j, i]) / bandwidth
            t_sum += ( 1 / bandwidth ) * KernelFunction( normalise )
        out += ( 1 / ( num_auctions * num_bidders ) ) * t_sum
    return out

def KernelFunction( value ):
    # Triweight
    out = ( 35 / 32 ) * ( 1 - value**2 )**3 * ( np.abs( value ) <= 1 )
    # Epanechikov
    #out = 0.75 * ( 1 - value**2 ) * ( np.abs( value ) <= 1 )
    # Standard Normal
    #out = ( 2 * np.pi )**(-0.5) * np.exp( -0.5 * value**2 )
    return out

# Estimate our Distributions over differing values for v for each n
maximum_bid = int(np.ceil(np.max(bids))) + 5
Gv = np.zeros( shape=(maximum_bid,) )
gv = np.zeros( shape=(maximum_bid,) )

# Conduct Kernel Density Distribution
for v in range( maximum_bid ):
    Gv[v] = EstimateG( v )
    gv[v] = Estimateg( v )

# Draw Graph of Cumulative Distribution
print( 'Outputting pdf of Bids ...' )
print( gv.shape )
print( gv )
print('')

# Draw Graph of Cumulative Distribution
print( 'Generating Plot of Distribution for pdf of Bids ...' )
plt.plot( gv )
plt.title('Non-Parametric Probability Density Function for Bids')
plt.xlabel('v')
plt.ylabel('Pr(Valuation=v)')
plt.show()
print('')

# Draw Graph of Cumulative Distribution
print( 'Outputting CDF of Bids ...' )
print( Gv.shape )
print( Gv )
print('')

# Draw Graph of Cumulative Distribution
print( 'Generating Plot of Distribution for CDF of Bids ...' )
plt.plot( Gv )
plt.title('Non-Parametric Cumulative Distribution Function for Bids')
plt.xlabel('v')
plt.ylabel('Pr(Valuation<=v)')
plt.show()
print('')

# Estimate FPSB Private Values
fpsb_private_values = ObtainFPSBPrivateValue()
print( 'Outputting FPSB Private Values ...' )
print( fpsb_private_values )
print('')

# Draw Graph of Cumulative Distribution
print( 'Generating Histogram of Distribution for FPSB Private Values ...' )
num_bins = 20
for i in range( num_bidders ):
    print( 'Graph [' + str(i+1) + ']' )
    N, bins, patches = plt.hist(fpsb_private_values[:, i], bins=num_bins)
    fracs = N / N.max()
    norm  = colors.Normalize( fracs.min(), fracs.max() )
    for ifrac, ipatch in zip( fracs, patches ):
        color = plt.cm.viridis( norm( ifrac ) )
        ipatch.set_facecolor( color )
    plt.title('Distribution of FPSB Private Values (bidder ' + str(i+1) + ')')
    plt.xlabel('Estimated FPSB Private Values')
    plt.ylabel('Frequency')
    plt.show()
print('')

# Conduct Kernel Density Distribution
maximum_value = int( np.ceil( np.max( fpsb_private_values ) ) ) + 5
fpsb_fu = np.zeros( (maximum_value,) )
fpsb_Fu = np.zeros( (maximum_value,) )
for v in range( maximum_value ):
    fpsb_fu[v] = ObtainFPSBf( v )
#normalise_min = np.min(fpsb_fu)
#normalise_max = np.max(fpsb_fu)
#for v in range( maximum_value ):
#    fpsb_fu[v] = ( fpsb_fu[v] - normalise_min ) / ( normalise_max - normalise_min )
fpsb_Fu[0] = fpsb_fu[0]
for v in range( maximum_value - 1 ):
    fpsb_Fu[v + 1] = fpsb_fu[v + 1] + fpsb_Fu[v]

print( 'Outputting pdf of Private Values  ...' )
print( fpsb_fu )
print('')

# Draw Graph of Cumulative Distribution
print( 'Generating Plot of Distribution for pdf of Private Values ...' )
plt.plot( fpsb_fu )
plt.title('Non-Parametric Probability Density Function for Private Values')
plt.xlabel('v')
plt.ylabel('Pr(Valuation=v)')
plt.show()
print('')

print( 'Outputting CDF of Private Values...' )
print( fpsb_Fu )
print('')

# Draw Graph of Cumulative Distribution
print( 'Generating Plot of Distribution for CDF of Private Values ...' )
plt.plot( fpsb_Fu )
plt.title('Non-Parametric Cumulative Distribution Function for Private Values')
plt.xlabel('v')
plt.ylabel('Pr(Valuation<=v)')
plt.show()
print('')

# First and Third Quartiles of Marginal Distribution F of our Private Values
pv_quantile = np.zeros( (2,) )
pv_quantile[0] = np.percentile( fpsb_Fu, 25 )
pv_quantile[1] = np.percentile( fpsb_Fu, 75 )
pv_label   = np.zeros( (2**4, 4) )
pv_matrix  = np.zeros( (2**4, 4) )
pv_results = np.zeros( (2**4, 4) )
pv_final   = np.zeros( (2**4,) )
m = 0

for i in range( 2 ):
    for j in range( 2 ):
        for k in range( 2 ):
            for l in range( 2 ):
                pv_label[m, 0] = 25 + 50 * i
                pv_label[m, 1] = 25 + 50 * j
                pv_label[m, 2] = 25 + 50 * k
                pv_label[m, 3] = 25 + 50 * l
                pv_matrix[m, :] = np.array( [pv_quantile[i], pv_quantile[j], pv_quantile[k], pv_quantile[l]] )
                for n in range( num_bidders ):
                    pv_results[m, n] = np.percentile( fpsb_Fu, pv_label[m, n] )
                a = np.reshape( np.array( fpsb_Fu[0] ), (1, 1) )
                b, c, d = a, a, a
                n = 1
                while fpsb_Fu[n] <= pv_results[m, 0]:
                    a = np.concatenate( (a, np.reshape( np.array( fpsb_Fu[n] ), (1, 1) ) ), axis=0 )
                    n += 1
                a = np.mean( a )
                n = 1
                while fpsb_Fu[n] <= pv_results[m, 1]:
                    b = np.concatenate( (b, np.reshape( np.array( fpsb_Fu[n] ), (1, 1) ) ), axis=0 )
                    n += 1
                b = np.mean(b)
                n = 1
                while fpsb_Fu[n] <= pv_results[m, 2]:
                    c = np.concatenate( (c, np.reshape( np.array( fpsb_Fu[n] ), (1, 1) ) ), axis=0 )
                    n += 1
                c = np.mean(c)
                n = 1
                while fpsb_Fu[n] <= pv_results[m, 3]:
                    d = np.concatenate( (d, np.reshape( np.array( fpsb_Fu[n] ), (1, 1) ) ), axis=0 )
                    n += 1
                d = np.mean(d)
                n = 1
                pv_final[m] = np.mean( [a, b, c, d] )
                print( '[' + str(m) +']' )
                print( 'u_pc: ' + str( pv_label[m, :] ) )
                print( 'u_i:  ' + str( pv_results[m, :] ) )
                print( 'Gu_i: ' + str( pv_final[m] ) )
                print( 'Fu_i: ' + str( pv_label[m, 0] * pv_label[m, 1] * pv_label[m, 2] * pv_label[m, 3] / 100**4 ) )
                print('')
                m += 1


print('')
print('***')
print('[EOF] Output Terminates')
print('***')
# EOF
