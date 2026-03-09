import math

# --- Defining (exact) SI constants ---
SPEED_OF_LIGHT = float(299_792_458)     # c, speed of light [m/s] (exact)
PLANCK_H_J_S = 6.626_070_15e-34             # h, Planck constant [J*s] (exact)
ELEMENTARY_CHARGE_C = 1.602_176_634e-19     # e, elementary charge [C] (exact)
BOLTZMANN_J_K = 1.380_649e-23               # k_B, Boltzmann constant [J/K] (exact)
AVOGADRO_1_MOL = 6.022_140_76e23            # N_A, Avogadro constant [1/mol] (exact)

# Common exact reference values
STANDARD_GRAVITY_M_S2 = 9.806_65            # g_n, standard gravity [m/s^2] (exact)
STANDARD_ATMOSPHERE_PA = float(101_325)     # atm, standard atmosphere [Pa] (exact)

PI = math.pi                                 # π [-]
HBAR_J_S = PLANCK_H_J_S / (2.0 * PI)         # ħ, reduced Planck constant [J*s]
ELECTRON_VOLT_J = ELEMENTARY_CHARGE_C        # 1 eV in joule [J] (exact)
FARADAY_C_MOL = AVOGADRO_1_MOL * ELEMENTARY_CHARGE_C     # F [C/mol]
GAS_CONSTANT_J_MOL_K = AVOGADRO_1_MOL * BOLTZMANN_J_K     # R [J/(mol*K)]
STEFAN_BOLTZMANN_W_M2_K4 = 5.670_374_419e-8              # σ [W/(m^2*K^4)] (exact)

# --- Electromagnetism ---
EPS0_F_M = 8.854_187_8188e-12                # ε0, vacuum permittivity [F/m]
MU0_N_A2 = 1.256_637_061_27e-6               # μ0, vacuum permeability [N/A^2]
VACUUM_IMPEDANCE_OHM = 376.730_313_412       # Z0, vacuum impedance [ohm]
COULOMB_CONSTANT_N_M2_C2 = 1.0 / (4.0 * PI * EPS0_F_M)   # k_e [N*m^2/C^2]

# --- Atomic / particle physics ---
U_KG = 1.660_539_068_92e-27                  # u (atomic mass constant) [kg]
ELECTRON_MASS_KG = 9.109_383_7139e-31        # m_e [kg]
PROTON_MASS_KG = 1.672_621_925_95e-27        # m_p [kg]
NEUTRON_MASS_KG = 1.674_927_500_56e-27       # m_n [kg]

FINE_STRUCTURE_ALPHA = 7.297_352_5643e-3     # α [-]
BOHR_RADIUS_M = 5.291_772_105_44e-11         # a0 [m]
RYDBERG_M_1 = 10_973_731.568_157             # R_inf [1/m]
HARTREE_ENERGY_J = 4.359_744_722_2060e-18    # E_h [J]

BOHR_MAGNETON_J_T = 9.274_010_0657e-24       # μ_B [J/T]
NUCLEAR_MAGNETON_J_T = 5.050_783_7393e-27    # μ_N [J/T]

# --- Gravitation ---
GRAVITATIONAL_CONSTANT_M3_KG_S2 = 6.674_30e-11  # G [m^3/(kg*s^2)]