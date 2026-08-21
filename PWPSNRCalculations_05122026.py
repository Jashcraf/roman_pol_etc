import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np

from corgisim import scene, instrument

PROJECT_DIR = Path.home() / "roman_pol_etc"
gain = 1626.595 
cr_rate = 0
N_COADDS = 3
single_texp = 28 
N_FRAMES = 7

# Load corgisim-generated scenes and fields
with open(PROJECT_DIR / "polarized_fields.pickle", "rb") as f:
    polarized_fields = pickle.load(f)

with open(PROJECT_DIR / "polarized_scenes.pickle", "rb") as f:
    polarized_scenes = pickle.load(f)

# Build up the interaction matrix
probe_labels = ["cos", "sin1", "sin2"]
prism_labels = ["POL0", "POL45"]
channel_labels = ["I0", "I90", "I45", "I135"]
weight = 1

# Make interaction matrix broadcast over polarization dimension
# Should be Nprism x Npix x Npix x 3 x 2 array, where 3 is probe index
# and 2 is the Real/Imaginary component. Remember to subtract off nominal field
# Modify polarized fields - start with POL0
nominal_field_unocculted = polarized_fields[("unprobed", False, "POL0")].sum(axis=(0,1))
print(f"nominal field shape = {nominal_field_unocculted.shape}")

# Normalize by max of |Field|, which is in sqrt(contrast) units
# nominal_field_unocculted is already complex (51, 51); take |.| over the whole image.
root_c_norm = np.abs(nominal_field_unocculted).max()
nominal_field_occulted_0 = polarized_fields[("unprobed", True, "POL0")].sum(axis=(0,1)) / root_c_norm
nominal_field_occulted_90 = polarized_fields[("unprobed", True, "POL0")].sum(axis=(0,1)) / root_c_norm
nominal_field_occulted_45 = polarized_fields[("unprobed", True, "POL45")].sum(axis=(0,1)) / root_c_norm
nominal_field_occulted_135 = polarized_fields[("unprobed", True, "POL45")].sum(axis=(0,1)) / root_c_norm

nom_fields = [
    (nominal_field_occulted_0, nominal_field_occulted_90),
    (nominal_field_occulted_45, nominal_field_occulted_135)
]

# Set up interaction matrix
interaction_matrix = np.zeros((len(channel_labels), *nominal_field_unocculted.shape, 3, 2))
differential_vector = np.zeros((len(channel_labels), *nominal_field_unocculted.shape, 3)) # stores one element per pair-wise probe
differential_variance = np.zeros((len(channel_labels), *nominal_field_unocculted.shape, 3)) # stores one element per pair-wise probe

# Label doesn't do anything here, it's basically a comment for why we have "offset"
for offset, label, nom_field in zip([0, 2], ["POL0", "POL45"], nom_fields):
    for j, probe_label in enumerate(probe_labels):

        # Define key
        key = (probe_label, 1, label)

        # Critically, subtracting off the unprobed field
        field_0 = polarized_fields[key].sum(axis=(0, 1)) / root_c_norm 
        field_0 -= nom_field[0] 
        
        field_90 = polarized_fields[key].sum(axis=(0, 1)) / root_c_norm 
        field_90 -= nom_field[1]

        # pol, spatial, spatial, probe, Real/Imaginary
        interaction_matrix[offset, :, :, j, 0] = field_0.real
        interaction_matrix[offset, :, :, j, 1] = field_0.imag

        interaction_matrix[offset + 1, :, :, j, 0] = field_90.real
        interaction_matrix[offset + 1, :, :, j, 1] = field_90.imag

# Compute difference images using corgisim detector simulations
# Set up corgisim detector
N_COADDS = int(N_COADDS)
N_FRAMES = int(N_FRAMES)


emccd_keywords ={'em_gain':gain,
                 'cr_rate':cr_rate}
detector = instrument.CorgiDetector(emccd_keywords,
                                    photon_counting=False)

# Grab probe state
I0s_p = []
I90s_p = []
I45s_p = []
I135s_p = []

I135s_m = []
I45s_m = []
I90s_m = []
I0s_m = []

# Get unocculted scene for contrast conversion
nominal_scene_unocculted = polarized_scenes[("unprobed", False, "POL0")]
t_unoc = 1e-5
sim_scene = detector.generate_detector_image(nominal_scene_unocculted, t_unoc)
unoc_0 = sim_scene.image_on_detector.data[0].astype(float) / t_unoc
plt.figure()
plt.imshow(unoc_0)
plt.colorbar()
plt.show()
unoc_90 = sim_scene.image_on_detector.data[1].astype(float) / t_unoc
unoc_I = unoc_0 + unoc_90
contrast_norm = unoc_I.max() # photons / s

for i, label in enumerate(probe_labels):
    for x in range(N_FRAMES):

        # Init intensities to co-add 
        I0_p = 0.
        I90_p = 0.
        I45_p = 0.
        I135_p = 0.
        
        I0_m = 0.
        I90_m = 0.
        I45_m = 0.
        I135_m = 0.

        for _ in range(N_COADDS):
            
            # Positive probe weights
            sim_scene_Q_p = polarized_scenes[(label, 1, "POL0")]
            sim_scene_U_p = polarized_scenes[(label, 1, "POL45")]
            
            # Re-sample scene from flux map
            sim_scene = detector.generate_detector_image(sim_scene_Q_p, single_texp)
            I0_p += sim_scene.image_on_detector.data[0].astype(float) / contrast_norm / single_texp
            I90_p += sim_scene.image_on_detector.data[1].astype(float) / contrast_norm / single_texp
            
            sim_scene = detector.generate_detector_image(sim_scene_U_p, single_texp)
            I45_p += sim_scene.image_on_detector.data[0].astype(float) / contrast_norm / single_texp
            I135_p += sim_scene.image_on_detector.data[1].astype(float) / contrast_norm / single_texp

            # Negative probe weights
            sim_scene_Q_m = polarized_scenes[(label, -1, "POL0")]
            sim_scene_U_m = polarized_scenes[(label, -1, "POL45")]
            
            # Re-sample scene from flux map
            sim_scene = detector.generate_detector_image(sim_scene_Q_m, single_texp)
            I0_m += sim_scene.image_on_detector.data[0].astype(float) / contrast_norm / single_texp
            I90_m += sim_scene.image_on_detector.data[1].astype(float) / contrast_norm / single_texp
            
            sim_scene = detector.generate_detector_image(sim_scene_U_m, single_texp)
            I45_m += sim_scene.image_on_detector.data[0].astype(float) / contrast_norm / single_texp
            I135_m += sim_scene.image_on_detector.data[1].astype(float) / contrast_norm / single_texp
            
        I0s_p.append(I0_p)
        I90s_p.append(I90_p)
        I45s_p.append(I45_p)
        I135s_p.append(I135_p)
        
        I0s_m.append(I0_m)
        I90s_m.append(I90_m)
        I45s_m.append(I45_m)
        I135s_m.append(I135_m)

    # Get mean and variance
    I0_p = np.nanmean(I0s_p, axis=0)
    I90_p = np.nanmean(I90s_p, axis=0)
    I45_p = np.nanmean(I45s_p, axis=0)
    I135_p = np.nanmean(I135s_p, axis=0)
    
    I0_m = np.nanmean(I0s_m, axis=0)
    I90_m = np.nanmean(I90s_m, axis=0)
    I45_m = np.nanmean(I45s_m, axis=0)
    I135_m = np.nanmean(I135s_m, axis=0)

    var_I0_p = np.nanvar(I0s_p, axis=0)
    var_I90_p = np.nanvar(I90s_p, axis=0)
    var_I45_p = np.nanvar(I45s_p, axis=0)
    var_I135_p = np.nanvar(I135s_p, axis=0)

    var_I0_m = np.nanvar(I0s_m, axis=0)
    var_I90_m = np.nanvar(I90s_m, axis=0)
    var_I45_m = np.nanvar(I45s_m, axis=0)
    var_I135_m = np.nanvar(I135s_m, axis=0)

    # store vector for inversion 
    differential_vector[0, :, :, i] = I0_p - I0_m  
    differential_vector[1, :, :, i] = I90_p - I90_m  
    differential_vector[2, :, :, i] = I45_p - I45_m  
    differential_vector[3, :, :, i] = I135_p - I135_m

    # Store variance for noise propagation
    differential_variance[0, :, :, i] = var_I0_p + var_I0_m  
    differential_variance[1, :, :, i] = var_I90_p + var_I90_m 
    differential_variance[2, :, :, i] = var_I45_p + var_I45_m 
    differential_variance[3, :, :, i] = var_I135_p + var_I135_m

# make consistent with PWP math
differential_vector /= 4

# Perform the least-squares inversion of the H matrix
Hinv = np.linalg.pinv(interaction_matrix)
state_vector = Hinv @ differential_vector[..., None]
state_vector = (state_vector[..., 0])

# Propagate standard deviation (contrast units) into E-fields
state_variance = np.abs(Hinv) @ np.sqrt(differential_variance[..., None])
state_variance = state_variance[..., 0]

if True:
    fig, axs = plt.subplots(figsize=[8, 4], ncols=2, nrows=4)
    for i in range(4):
        if i == 0:
            axs[i, 0].set_title("Real Part")
            axs[i, 1].set_title("Imag Part")

        # Aesthetics
        axs[i, 0].set_ylabel(channel_labels[i])
        
        im = axs[i, 0].imshow(state_vector[i, ..., 0] / state_variance[i, ..., 0], cmap="viridis", vmax=None, vmin=None)
        div = make_axes_locatable(axs[i, 0])
        cax = div.append_axes("right", size="7%", pad="2%")
        fig.colorbar(im, cax=cax)
        
        im = axs[i, 1].imshow(state_vector[i, ..., 1] / state_variance[i, ..., 1], cmap="plasma", vmax=None, vmin=None)
        div = make_axes_locatable(axs[i, 1])
        cax = div.append_axes("right", size="7%", pad="2%")
        fig.colorbar(im, cax=cax)

# compute electric field from probes
i_sel = 0
print(f"state vector shape = {state_vector.shape}")
E_0 = state_vector[i_sel, ..., 0] + 1j * state_vector[i_sel, ..., 1]
I_0_est = np.abs(E_0) ** 2 
# I_0_est /= 4

peak_flux_est = I_0_est.max()

# Simulate I0 image is unoc_0
nominal_scene_occulted = polarized_scenes[("unprobed", True, "POL0")]

occulted_I0 = []
for x in range(N_FRAMES):
    
    oc_0 = 0
    
    for y in range(N_COADDS):
        sim_scene = detector.generate_detector_image(nominal_scene_occulted, single_texp)
        oc_0 += sim_scene.image_on_detector.data[0].astype(float) / contrast_norm / single_texp
    
    occulted_I0.append(oc_0)

oc_0 = np.nanmean(occulted_I0, axis=0)
peak_flux_nom = oc_0.max()

# max norm
# oc_0 /= peak_flux_nom
# I_0_est /= peak_flux_est

ims = [
    oc_0,
    I_0_est,
    oc_0 - I_0_est
]

titles = [
    "Image",
    "Estimated \n Coherent Intensity",
    "Estimated \n Incoherent Intensity"
]

# Show the estimated coherent and incoherent parts - I90
fig, axs = plt.subplots(figsize=[10, 4], ncols=3, nrows=1)

from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
cmap = "inferno"
for i in range(3):
    
    norm = LogNorm(vmin=1e-9, vmax=1e-7)

    if i == 2:
        norm = None
        cmap = 'RdBu_r'
    
    ax = axs[i]
    ax.set_title(titles[i])
    im = ax.imshow(ims[i], norm=norm, cmap=cmap)
    
    div = make_axes_locatable(ax)
    cax = div.append_axes("right", size="5%", pad="1%")
    fig.colorbar(im, cax=cax)

plt.show()
