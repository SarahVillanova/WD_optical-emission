import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.interpolate import interp1d

# Plotting parameters
params = {
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'axes.labelweight': 'heavy',
    'legend.fontsize': 20,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'text.usetex': True,
    'figure.figsize': [8,7],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.serif': ['Times'],
    'font.weight': 'heavy',
    'lines.linewidth': 2
}
plt.rcParams.update(params)

# Physical parameters
d = 1.3e3  # distance in pc
delta_d = 0.4 * d  # distance uncertainty
ext = 0.77  # extinction in mag
delta_ext = 0.03  # extinction uncertainty
mag_threshold = 24.4

# Column names for the data files
column_names = [
    'Teff', 'logL_Lsun', 'tcool_Gyr', 'logg', 'radius_Rsun',
    'G3', 'Bp3', 'Rp3', 'u', 'g', 'r', 'i', 'z',
    'g2', 'r2', 'i2', 'z2', 'y'
]

# Model files and corresponding masses
mass_file_pairs = [
    (1.10, "110Hrich_0.001.dat"),
    (1.16, "116Hrich_0.001.dat"),
    (1.23, "123Hrich_0.001.dat"),
    (1.29, "129Hrich_0.001.dat"),
    (1.31, "t131CO_R.dat"),
    (1.33, "t133CO_R.dat"),
    (1.35, "t135CO_R.dat"),
    (1.37, "t137CO_R.dat"),
    (1.382, "t1382CO_R.dat")
]

# Calculate magnitude uncertainty
sigma_d = (5 / np.log(10)) * (delta_d / d)
sigma_G3_rel = np.sqrt(sigma_d**2 + delta_ext**2)

first_detection_ages = []
upper_ages = []
lower_ages = []
masses_used = []

for mass, file_path in mass_file_pairs:
    try:
        table = Table.read(file_path, format='ascii.no_header')
        
        # Rename columns
        for old, new in zip(table.colnames, column_names):
            table.rename_column(old, new)

        # Compute apparent G3 magnitude
        table['G3_rel'] = table['G3'] + 5 * np.log10(d) - 5 + ext

        # Get model age limits
        min_age = np.min(table['tcool_Gyr'])
        max_age = np.max(table['tcool_Gyr'])
        
        # Check if all magnitudes are brighter than threshold
        if np.all(table['G3_rel'] < mag_threshold):
            # For always detectable cases (like 1.382)
            age_central = min_age
            age_lower = min_age
            # Find when it would become undetectable
            rev_interp = interp1d(table['G3_rel'], table['tcool_Gyr'], 
                                bounds_error=False, fill_value=max_age)
            age_upper = rev_interp(mag_threshold + sigma_G3_rel)
        else:
            # Normal case
            interp_func = interp1d(table['G3_rel'], table['tcool_Gyr'],
                                 bounds_error=False, fill_value=None)
            
            age_central = interp_func(mag_threshold)
            age_lower = interp_func(mag_threshold - sigma_G3_rel)
            age_upper = interp_func(mag_threshold + sigma_G3_rel)
            
            # Handle out-of-bounds cases
            age_lower = min_age if np.isnan(age_lower) else age_lower
            age_upper = max_age if np.isnan(age_upper) else age_upper
            age_central = min_age if np.isnan(age_central) else age_central

        # Store results
        first_detection_ages.append(age_central)
        upper_ages.append(age_upper)
        lower_ages.append(age_lower)
        masses_used.append(mass)
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        continue

# Convert to numpy arrays
masses_arr = np.array(masses_used)
ages_arr = np.array(first_detection_ages)
upper_arr = np.array(upper_ages)
lower_arr = np.array(lower_ages)

# Find 1.382 index and its age for y-axis limit
idx_1382 = np.where(masses_arr == 1.382)[0][0]
y_max = upper_arr[idx_1382] * 1.05  # 5% padding above 1.382 upper limit

# Plotting
plt.figure(figsize=(8,7))
plt.plot(masses_arr, ages_arr, '-o', markersize=8, linewidth=2.5,
         label='Cooling Age', color='maroon')
plt.fill_between(masses_arr, lower_arr, upper_arr,
                alpha=0.2, color='maroon', label='Uncertainty')

plt.xlabel(r'Mass ($M_{\odot}$)', labelpad=10)
plt.ylabel(r'Minimum Cooling Age (Gyr)', labelpad=10)
plt.grid(True, which='both', linestyle='--', alpha=0.7)
plt.xlim(min(masses_arr)-0.005, max(masses_arr)+0.005)
plt.ylim(0, 3.2)  # Cut y-axis at 1.382's upper limit
plt.savefig("cooling.png", bbox_inches='tight', dpi=300)
print(ages_arr)
print(masses_arr)
