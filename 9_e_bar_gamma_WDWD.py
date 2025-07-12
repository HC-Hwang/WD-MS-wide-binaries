import numpy as np

import astropy.units as u
from astropy.table import Table, Column, MaskedColumn, join, vstack
from astropy.io import ascii, fits
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse_single_table

from scipy import stats, interpolate
import scipy

from tqdm import tqdm, tqdm_notebook
import copy

from numpy import cos, sin
import datetime
import sys

def vr_angle_pdf_gaussian(vr_angle, vr_angle_error, vr_angle_list=np.arange(0., 181., 1.)):
    
    gaussian = scipy.stats.norm(vr_angle, vr_angle_error).pdf(vr_angle_list) 
    return gaussian / np.sum(gaussian)


def log_probability(alpha,
                    p_gamma_bar_e_grid,
                    vr_angle_pdf_list,
                    e_dense_list
                   ):
    
    log_p = 0.
    
    if alpha < -0.99:
        return -np.inf

    p_e_bar_a = (e_dense_list)**(alpha)
    
    #normalize numerically. Analytic form does not work well for negative alpha
    p_e_bar_a = p_e_bar_a / np.sum(p_e_bar_a) 
    
    for i in range(len(vr_angle_pdf_list)):
        if i % 10000 == 0:
            print(' '*5, datetime.datetime.now())
            print('     pair_i: %d/%d' %(i, len(vr_angle_pdf_list)))
        
        term0 = p_gamma_bar_e_grid * vr_angle_pdf_list[i][:, None] * p_e_bar_a[None, :]
        
        log_p += np.log(np.sum(term0))
    
    return log_p

def derive_p_alpha_bar_gamma(gamma, gamma_error, 
        e_dense_list = np.arange(1e-3, 1., 0.01),
        vr_angle_list_to_integrate = np.arange(0., 181., 1),
        alpha_list = np.arange(-1.+0.01, 3., 0.01)
    ):
    """
    gamma: N-dimension array, v-r angles in deg
    gamma_error: N-dimension array, uncertainties of v-r angles in deg
    """
    
    p_gamma_bar_e_grid = p_gamma_bar_e(e_dense_list,vr_angle_list_to_integrate)
    
    #pre-compute p(v_true|v_obs) for each pair
    
    print('-'*10)
    print(datetime.datetime.now())
    print('Computing vr_angle_pdf_list...')
    sys.stdout.flush()

    vr_angle_pdf_list = []
    N_binary = len(gamma)
    for i in range(N_binary):
        vr_angle_pdf_list.append(vr_angle_pdf_gaussian(
            gamma[i], gamma_error[i], 
            vr_angle_list=vr_angle_list_to_integrate))


   
    print('-'*10)
    print(datetime.datetime.now())
    print('Compute probability for each alpha in the list')
    sys.stdout.flush()

    log_probability_list = []
    for i, alpha in enumerate(alpha_list):
        
        print('-'*10)
        print(datetime.datetime.now())
        print('alpha: %d/%d' %(i, len(alpha_list)))

        log_probability_list.append(log_probability(alpha, p_gamma_bar_e_grid, vr_angle_pdf_list, e_dense_list))
        
    return np.array(log_probability_list), alpha_list


sep_0 = float(sys.argv[1])
sep_1 = float(sys.argv[2])

output_path = './9_results_WDWD/'

print('Command line:')
print(str(sys.argv))

print('-'*10)
print(datetime.datetime.now())
print('Reading wide binary table...')
sys.stdout.flush()

#wb_table = Table.read('../../2021_0_GaiaEDR3_LAMOST_wide_binaries/1_LAMOST_wide_binaries/catalogs/KE_all_columns_catalog.fits.gz')

wb_table = Table.read('../2021_6_wide_binary_eccentricity/16_address_referee_report/16_full_individual_wide_binaries.fits')

print('-'*10)
print(datetime.datetime.now())
print('Computing v-r angles for wide binaries...')
sys.stdout.flush()


print('-'*10)
print(datetime.datetime.now())
print('Load pre-computed p_gamma_bar_e grid...')
e_list, vr_angle_list_center, hist_list = np.load('../2021_6_wide_binary_eccentricity/2_Gaia_measurements/2_results_N_1e6/data.npy', allow_pickle=True)
p_gamma_bar_e = interpolate.interp2d(e_list, vr_angle_list_center, hist_list.T, kind='cubic')

t = (
    (wb_table['R_chance_align'] < 0.1) *
    (wb_table['dpm_over_error'] > 3) *
    (wb_table['parallax1'] > 5) *
    (wb_table['sep_AU'] > sep_0) *
    (wb_table['sep_AU'] < sep_1) *
    (wb_table['pairdistance'] * 3600. > 1.5) *
    (wb_table['binary_type'] == 'WDWD')
)


log_probability_list, alpha_list = derive_p_alpha_bar_gamma(
                            wb_table[t]['vr_angle'],
                            wb_table[t]['vr_angle_error'],
                                                )


np.save(output_path + 'sep_%d_%d'%(sep_0, sep_1), [alpha_list, log_probability_list])
print('-'*10)
print(datetime.datetime.now())
print('Save the results as: ' + output_path + 'sep_%d_%d.npy'%(sep_0, sep_1))
print('Program ends successfully')

