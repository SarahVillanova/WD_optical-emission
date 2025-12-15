import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

params = {'backend': 'agg',
          'axes.labelsize': 20,  
          'axes.titlesize': 24,
          'axes.labelweight': 'heavy',
          'legend.fontsize': 20,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': True,
          'figure.figsize': [8,6],
          'figure.dpi': 300,
          'savefig.dpi': 300,
          'font.family': 'serif',
          'font.serif': ['Times'],
          'font.weight': 'heavy',
          'lines.linewidth': 2
}
plt.rcParams.update(params)

table = Table.read("gaiaedr3_wd_main.fits")
df = table.to_pandas()

df['volume'] = 4/3 * np.pi * (df['r_med_geo'] ** 3)
vtot = 4/3 * np.pi * (200 ** 3)
V_total = (4/3) * np.pi * (200 ** 3)
vlim_values = {} 
V_lim = [0.048,0.17,0.27]
chisq_df = df[['chisq_H', 'chisq_He', 'chisq_mixed']].replace(0, np.nan).fillna(np.inf)
df['best_model'] = chisq_df.idxmin(axis=1)

df['mass'] = np.nan
df.loc[df['best_model'] == 'chisq_H', 'mass'] = df['mass_H']
df.loc[df['best_model'] == 'chisq_He', 'mass'] = df['mass_He']
df.loc[df['best_model'] == 'chisq_mixed', 'mass'] = df['mass_mixed']

df['abs_G_norm'] = 1.0

plt.figure(figsize=(8,6))
V_lim = [0.048, 0.17, 0.27]

for i in range(8, 18):
    mask = (
        (df['Pwd'] > 0.9) &
        (df['r_med_geo'] <= 200) &
        (df['absG'] > i - 0.5) &
        (df['absG'] <= i + 0.5) 
    )
    normal = df[mask]
    volume_sorted = np.sort(normal['volume'].values)
    cdf_normal = np.arange(1, len(volume_sorted) + 1) / len(volume_sorted) * 100

    plt.plot(volume_sorted/vtot, cdf_normal)

    if i in [15, 16, 17]:
        v_idx = i - 15  
        vlim_val = V_lim[v_idx] * vtot
        vlim_values[i] = vlim_val

        plt.axvline(vlim_val/vtot, color='black', linestyle='--', alpha=0.7)
        plt.text(vlim_val/vtot, 5, f'V_lim {i}', rotation=90,
                 verticalalignment='bottom', color='black')

plt.xlabel('Volume [pc³]')
plt.ylabel('Cumulative \%')
plt.title('Cumulative Distribution vs. Volume by M_G bin')
plt.legend(title="Absolute G Magnitude bins")
plt.grid(True)
plt.tight_layout()
plt.savefig("volume_thing.png")



filtered = df[
    (df['Pwd'] > 0.9) &
    (df['absG'].notna()) &
    (df['r_med_geo'] <= 200)
].copy()

# --- Apply V_lim volume cuts & weighting ---
def apply_vlim_weights(df, vlim_values, vtot):
    """
    For stars in magnitude bins 15, 16, 17:
      - keep only stars with volume <= vlim_values[i]
      - assign weight = vtot / vlim_values[i]
    For all other stars: weight = 1
    """
    df = df.copy()
    df["weight"] = 1.0

    for i in [15, 16, 17]:
        if i in vlim_values:
            vlim_val = vlim_values[i]
            mask_bin = (df["absG"] > i - 0.5) & (df["absG"] <= i + 0.5)
            # Apply cut
            df.loc[mask_bin & (df["volume"] > vlim_val), "weight"] = 0
            # Apply weight
            df.loc[mask_bin & (df["volume"] <= vlim_val), "weight"] = vtot / vlim_val
    return df

# Apply weighting to your filtered dataset
weighted_df = apply_vlim_weights(filtered, vlim_values, vtot)

# --- Define subpopulations again ---
very_massive = weighted_df[weighted_df['mass'] > 1.1]
massive = weighted_df[weighted_df['mass'] > 1.05]
normal = weighted_df[weighted_df['mass'] <= 1.4]

# --- Sort for CDFs ---
very_massive_sorted = very_massive.sort_values('absG')
massive_sorted = massive.sort_values('absG')
normal_sorted = normal.sort_values('absG')

# --- Compute weighted CDFs ---
g_very = very_massive_sorted['absG'].values
cdf_very = np.cumsum(very_massive_sorted["weight"].values) / very_massive_sorted["weight"].sum() * 100

g_massive = massive_sorted['absG'].values
cdf_massive = np.cumsum(massive_sorted["weight"].values) / massive_sorted["weight"].sum() * 100

g_normal = normal_sorted['absG'].values
cdf_normal = np.cumsum(normal_sorted["weight"].values) / normal_sorted["weight"].sum() * 100


def absolute_to_aparent(g_abs):
    return g_abs + 5*np.log10(1300) - 5 + 0.83

def aparent_to_absolute(g_app):
    return g_app - (5*np.log10(1300) - 5 + 0.83)

# --- Example data ---

# --- Plot ---
fig, ax_f = plt.subplots()

ax_f.plot(g_normal, cdf_normal, color='maroon')
ax_f.set_xlim(10, 16.5)

# Add secondary x-axis (top)
ax_top = ax_f.secondary_xaxis(
    'top',
    functions=(absolute_to_aparent, aparent_to_absolute)
)
ax_top.set_xlabel("$m_G$ for Gleam-X")

# Labels
ax_f.set_xlabel("$M_G$")
ax_f.set_ylabel("CDF (\%)")

plt.grid(True)
plt.tight_layout()
plt.savefig("cdf.png")