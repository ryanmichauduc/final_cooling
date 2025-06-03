import math
import os
import subprocess
import time

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from tqdm import *
import random

# The fields expected in the track files
FIELDS = [
    "x",
    "y",
    "z",
    "Px",
    "Py",
    "Pz",
    "t",
    "PDGid",
    "EventID",
    "TrackID",
    "ParentID",
    "Weight",
    "Bx",
    "By",
    "Bz",
    "Ex",
    "Ey",
    "Ez",
    "ProperTime",
    "PathLength",
    "PolX",
    "PolY",
    "PolZ",
    "InitX",
    "InitY",
    "InitZ",
    "InitT",
    "InitKE",
]

MUON_MASS = 105.6583755  # There's got to be a better way than this...
# MUON_MASS = 105.65836668  # This is the rest mass that Distribution_optics.m uses
C = 299792458

def read_trackfile(filepath):
    """
    Reads G4Beamline output into a Pandas database

    :param filepath: filepath to read
    :return: pandas dataframe of the track file
    """
    # explicit index_col=False apparently required for some files
    return pd.read_csv(filepath, sep=" ", comment="#", names=FIELDS, index_col=False)


def write_trackfile(tracks, filepath, comment=""):
    """
    Creates a trackfile in the #BLTrackFile2 format

    :param tracks: pandas dataframe of particle tracks to write
    :param filepath: filepath to write to
    :param comment: user comment on the file (usually a title)
    """
    with open(filepath, "w+") as file:
        file.write(f"#BLTrackFile2 {comment}\n")
        file.write("#" + " ".join(FIELDS) + "\n")
        tracks.to_csv(file, sep=" ", header=False, index=False, lineterminator="\n")


def calc_params(x, xp, delta, normalization=1):
    """Calculates Courant-Snyder parameters and emittance in a transverse dimension

    Implements the formulas at https://nicadd.niu.edu/~syphers/tutorials/analyzeTrack.html#analyzing-a-distribution.
    This is only correct for the x and y (transverse) axes, z (longitudinal) is different.

    Primarily a helper method for `calc_all_params`

    :param x: numpy array of positions, centered around ideal
    :param xp: numpy array of the derivative dx/dz
    :param delta: numpy array of the relative momentum deviation (in z direction)
    :param normalization: factor p/mc for normalized emittance. Can be left empty for non-normalized

    :return: a tuple consisting of (emittance, beta, gamma, alpha, dispersion, derivative of dispersion)
    """

    # Center x and xp
    x = x - np.mean(x)
    xp = xp - np.mean(xp)

    # G4Beamline gives all lengths in mm so that's what we assume here
    mean_x2 = np.mean(np.square(x))  # mm^2
    mean_xp2 = np.mean(np.square(xp))  # dimensionless
    mean_xxp = np.mean(x * xp)  # mm
    mean_d2 = np.mean(np.square(delta))  # dimensionless
    mean_xd = np.mean(x * delta)  # mm
    mean_xpd = np.mean(xp * delta)  # dimensionless

    eb = mean_x2 - np.square(mean_xd) / mean_d2  # mm^2
    ey = mean_xp2 - np.square(mean_xpd) / mean_d2  # dimensionless
    ea = -mean_xxp + mean_xpd * mean_xd / mean_d2  # mm
    e = np.sqrt(eb * ey - np.square(ea))  # mm

    d = mean_xd / mean_d2  # mm
    dp = mean_xpd / mean_d2  # dimensionless

    return (
        e * normalization,  # mm
        eb / e / 1000,  # mm --> m
        ey / e * 1000,  # 1/mm --> 1/m
        ea / e,  # dimensionless
        d / 1000,  # mm --> m
        dp,  # dimensionless
    )


def calc_all_params(df):
    """Calculates Courant-Snyder parameters in transverse directions, and emittance in longitudinal direction

    :param df: pandas dataframe of particles, parsed from G4Beamline output

    :return: tuple of x and y parameters, each consisting of (emittance, beta, gamma, alpha, D, D'), and the z-emittance
    """
    total_momentum = p_total(df)  # MeV/c
    mean_total_momentum = np.mean(total_momentum)
    delta = (total_momentum - mean_total_momentum) / mean_total_momentum

    # momentum is given as MeV*c and mass is given in MeV, so p/mc can be written simply as p/m
    beta = rel_beta(total_momentum)
    mean_time = np.mean(df["t"])
    t = df["t"] - mean_time
    # t is ns, C is m/s, result needs to be in mm, so overall factor of 10^-6
    z_pos = -C * beta * t * 1e-6

    x_params = calc_params(
        df["x"],
        df["Px"] / total_momentum,
        delta,
        normalization=mean_total_momentum / MUON_MASS,
    )
    y_params = calc_params(
        df["y"],
        df["Py"] / total_momentum,
        delta,
        normalization=mean_total_momentum / MUON_MASS,
    )
    z_emit = np.mean(beta) * np.std(z_pos) * np.std(df["Pz"]) / MUON_MASS

    return x_params, y_params, z_emit


def p_total(df):
    """
    Computes total momentum

    :param df: pandas dataframe of particles
    :return: Series representing the total momentum of the particles
    """
    return np.sqrt(
        np.square(df["Px"]) + np.square(df["Py"]) + np.square(df["Pz"])
    )


def emittances(df):
    """
    Calculates emittances of a distribution

    Simple wrapper on calc_all_params, filtering to just the emittances

    :param df: pandas dataframe of particles, parsed from G4Beamline output
    :return: tuple of emittances in x,y,z directions
    """
    x_params, y_params, z_emit = calc_all_params(df)
    return x_params[0], y_params[0], z_emit


def str_params(params):
    """
    Represents the parameters tuple in a convenient format for printing

    :param params: Tuple of (emittance, beta, gamma, alpha, D, D') as produced by calc_params
    :return: string representation of the parameters
    """
    return (
        f"emit  = {params[0]} mm\n"
        f"beta  = {params[1]} m\n"
        f"gamma = {params[2]} 1/m\n"
        f"alpha = {params[3]}\n"
        f"D     = {params[4]} m\n"
        f"D'    = {params[5]}"
    )


def print_all_params(df):
    """
    Prints parameters of a distribution (mostly for debugging)

    :param df: The distribution to summarize
    """
    x_params, y_params, z_emit = calc_all_params(df)
    print("-----------------------------")
    print("Twiss parameters for X")
    print(str_params(x_params))
    print()
    print("Twiss parameters for Y")
    print(str_params(y_params))
    print()
    print("Z-emittance: ", z_emit, "mm")
    total_momentum = p_total(df)
    beta = rel_beta(total_momentum)
    print("Z std:", C * np.mean(beta) * np.std(df["t"]) * 1e-6, "mm")
    print("p std:", np.std(total_momentum), "MeV/c")
    print("Mean momentum:", np.mean(total_momentum), "MeV/c")
    print("-----------------------------")


def gen_distribution(
        x_params, y_params, p_mean, p_std, z_std=None, z_emit=None, N=12000, particle_id=-13, seed=None
):
    """Generates a distribution with given parameters, using formulas in this link:
    https://nicadd.niu.edu/~syphers/tutorials/analyzeTrack.html#creating-a-distribution

    Generated distribution has z=0, with z_std reflected in time values.

    Generated distribution will also be centered at x=0, y=0, t=0. Displace as needed.

    Note that gamma is not part of the tuple, as it is redundant with alpha and beta.
    Note also that this tuple is given in a different order than `calc_params` outputs.

    :param x_params: Courant-Snyder parameters for the x-axis, given as a tuple of (beta, alpha, emittance, D, D')
    :param y_params: Courant-Snyder parameters for the y-axis, given as a tuple of (beta, alpha, emittance, D, D')
    :param p_mean: The mean momentum of the beam
    :param p_std: The standard deviation of total momentum
    :param z_std: The standard deviation of z location. Leave as None to infer
    :param z_emit: Emittance in z. Required if z_std is not given.
    :param N: Number of events to generate
    :param particle_id: PDGid of the particles to generate. Default is -13 representing a positive muon
    :param seed: Seed for the RNG (leave as None for random seed)
    """
    # Seed the RNG
    random = np.random.default_rng(seed)

    # Infer as necessary
    if z_std is None:
        z_std = infer_z_std(p_mean, p_std, z_emit)

    # Generate deltas
    delta = random.normal(0, p_std / p_mean, N)

    def gen_axis(b, a, e, d, dp):
        # Write b, d in mm instead of m
        b = b * 1000
        d = d * 1000

        sig_x = np.sqrt(b * (e * MUON_MASS / p_mean))  # de-normalize the emittance
        x0 = random.normal(0, sig_x, N)
        p0 = random.normal(0, sig_x, N)
        xp0 = (p0 - a * x0) / b
        x_ = x0 + d * delta
        xp_ = xp0 + dp * delta
        return x_, xp_

    x, xp = gen_axis(*x_params)
    y, yp = gen_axis(*y_params)
    zp = np.sqrt(1 - np.square(xp) - np.square(yp))
    p = p_mean * (1 + delta)

    # Translate z_std into a time std
    beta = rel_beta(p_mean)
    t_std = z_std / (C * beta * 1e-6)
    t = random.normal(0, t_std, N)

    result = pd.DataFrame(columns=FIELDS)
    result["x"] = x
    result["y"] = y
    result["z"] = 0
    result["Px"] = xp * p
    result["Py"] = yp * p
    result["Pz"] = zp * p
    result["t"] = t
    result["PDGid"] = particle_id
    result["EventID"] = np.arange(1, N + 1)
    result["TrackID"] = 1
    result["ParentID"] = 0
    result["Weight"] = 1
    # There are other fields, but leaving them empty is ok
    return result


def rel_beta(p):
    """Calculates relativistic beta

    :param p: momentum, in MeV/c (can be a numpy array)
    :return: beta factor (v/c)
    """
    return np.sqrt(1 - 1 / (1 + np.square(p / MUON_MASS)))


def remove_dispersion(df):
    """Simulates distribution after dispersion is removed"""
    df = df.copy(deep=True)
    total_momentum = p_total(df)  # MeV/c
    mean_total_momentum = np.mean(total_momentum)
    delta = (total_momentum - mean_total_momentum) / mean_total_momentum

    x_params, y_params, _ = calc_all_params(df)
    _, _, _, _, xD, xDp = x_params
    _, _, _, _, yD, yDp = y_params
    df["x"] = df["x"] - xD * delta * 1000
    df["y"] = df["y"] - yD * delta * 1000
    df["Px"] = df["Px"] - xDp * delta * total_momentum
    df["Py"] = df["Py"] - yDp * delta * total_momentum
    return df


def z_prop(df, z):
    """
    Approximates the propagation of particles down a drift channel, by adding to the

    :param df: initial particle distribution
    :param z: z-coordinate that the particles will be propagated to
    :return: distribution after drift
    """
    betas = rel_beta(df["Pz"])
    t_diff = (z - df["z"]) / (C * betas) * 1e6
    result = df.copy(deep=True)
    result["z"] = z
    result["t"] = result["t"] + t_diff
    return result


def df_calc(df):
    """
    Augments a pandas dataframe representing a particle distribution with calculated columns:
    "P" - total momentum
    "delta" - relative deviation from mean in total momentum
    "b_factor" - relativistic beta factor (v/c)

    :param df: the dataframe to augment
    """
    df["P"] = np.sqrt(np.square(df["Px"]) + np.square(df["Py"]) + np.square(df["Pz"]))
    mean_total_momentum = np.mean(df["P"])
    df["delta"] = (df["P"] - mean_total_momentum) / mean_total_momentum
    df["b_factor"] = rel_beta(df["P"])


def max_length(half_angle, height=10):
    """
    Calculates the maximum length geometrically possible for a particular angle. (This is specific to `G4_FinalCooling.g4bl`)

    :param half_angle: the half-angle being tested
    :param height: height of the wedge (default is the default in the file)
    :return: maximum value of `absLEN3` to not cause an error
    """
    return math.tan(half_angle * math.pi / 180) * height


def infer_z_std(p_mean, p_std, l_emittance):
    """
    Helper method to convert from longitudinal emittance to z std

    :param p_mean: Mean momentum (MeV/c)
    :param p_std: Standard deviation of momentum (MeV/c)
    :param l_emittance: Emittance in longitudinal direction (mm)
    :return: the z std corresponding to the given parameters
    """

    return l_emittance * MUON_MASS / (rel_beta(p_mean) * p_std)


def run_g4beam(df, filename, multioutput=None, debug=False, logging=False, **kwargs):
    import os, subprocess, time, random

    os.makedirs("temp", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    ident = str(time.time()).replace(".", "_") + "_" + str(random.randrange(256))
    in_filename = f"temp/in_{ident}.txt"
    out_prefix = f"temp/out_{ident}"
    out_filename = out_prefix + ".txt"
    log_filename = f"logs/log_{ident}.txt"

    write_trackfile(df, in_filename)

    command = [
        "/Applications/G4beamline-3.08.app/Contents/MacOS/g4bl",
        filename,
        f"beamfile={in_filename}",
        f"outname={out_prefix}",
        f"nparticles={len(df)}"
    ] + [f"{k}={v}" for k, v in kwargs.items()]

    if debug:
        print(" ".join(command))
        return

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if os.path.exists(in_filename):
            os.remove(in_filename)

    if multioutput:
        results = []
        for suffix in multioutput:
            fname = f"{out_prefix}-{suffix}.txt"
            results.append(read_trackfile(fname))
            if os.path.exists(fname):
                os.remove(fname)
        return results

    # Single-output
    try:
        result = read_trackfile(out_filename)
    except Exception as e:
        if os.path.exists(out_filename):
            os.remove(out_filename)
        raise RuntimeError(f"Failed to read G4Beamline output '{out_filename}': {e}")
    else:
        if os.path.exists(out_filename):
            os.remove(out_filename)
        return result


def run_distribution(df, length, half_angle, base_length=None, debug=False, filename="G4_FinalCooling_auto.g4bl",
                     axis=0, **kwargs):
    """
    Runs a given distribution through the wedge.
    This function is specific to the experiment contained in `G4_FinalCooling_auto.g4bl`

    If a height of N times offset is desired, use base length at least N*length

    :param df: The distribution to run
    :param length: Length of wedge at beam centerline
    :param half_angle: Half of the wedge angle
    :param base_length: The length of base to target. If unspecified, uses twice the offset.
    :param debug: If set to True, prints a parameter string that can be copied into G4Beamline, returns nothing
    :param filename: File to run. Should almost always be left as default, but can be changed
    :param axis: Axis 0 (x) or 1 (y)
    :return: Simulation results, or None if debug=True
    """
    offset = length / (math.tan(half_angle * math.pi / 180) * 2)
    if base_length is None:
        base_length = 2 * offset
    parameters = dict(
        abshalfangle3=str(half_angle),
        absoffset3=str(offset),
        abshgt=str(offset * base_length / length + 0.1),
        absLEN3=str(base_length),
        wedgeAxis=axis,
        **kwargs
    )
    if debug:
        print(" ".join(f"{x}={y}" for x, y in parameters.items()))
        write_trackfile(df, "particles_before.txt")
    else:
        return run_g4beam(
            df,
            filename=filename,
            **parameters
        )


def cut_outliers(distribution, sigma=4):
    """Removes outliers in position and momentum from a distribution

    :param distribution: the distribution to cut
    :param sigma: how many standard deviations from the center should be considered an outlier. Default of 5 is good.
    :return: the distribution with outliers removed
    """
    result = distribution
    for col in ["x", "y", "t", "Px", "Py", "Pz"]:
        mean = np.mean(distribution[col])
        std = np.std(distribution[col])
        result = result[(result[col] <= mean + sigma * std) & (result[col] >= mean - sigma * std)]
    return result


def remove_transverse(df):
    """Removes transverse momentum from a distribution

    :param df: Distribution to remove
    :return: Copy of df with Px and Py set to 0
    """
    df = df.copy(deep=True)
    df["Px"] = 0
    df["Py"] = 0
    return df


def recombine_transverse(df, t_df):
    """Adds transverse momentum from one distribution to another

    :param df: Distribution to modify
    :param t_df: Distribution with transverse momenta to add. Particles must be a superset of the particles of df
    :return: Copy of df with transverse momentums in t_df added
    """
    df = df.copy(deep=True)
    # Eventually, I might set it so all dataframes are indexed by eventid, so this is unnecessary?
    df.set_index("EventID", inplace=True, drop=False)
    t_df.set_index("EventID", inplace=True, drop=False)
    df["Px"] += t_df["Px"]
    df["Py"] += t_df["Py"]
    return df


def recenter_t(df):
    """Modifies t values of a distribution to center on t=0

    :param df: Distribution to modify
    :return: df with mean(t) = 0
    """
    mean_t = np.mean(df["t"])
    df["t"] -= mean_t
    return df


def cut_pz(df, tails=0.15):
    """Cuts a given amount of the tails of a distribution in the pz axis

    :param df: Distribution to cut
    :param tails: Total amount to cut (split across upper and lower tails)
    :return: Distribution with cuts made
    """
    tails /= 2  # Remove this much from each side
    lowest = np.quantile(df["Pz"], tails)
    highest = np.quantile(df["Pz"], 1 - tails)
    return df[(df["Pz"] >= lowest) & (df["Pz"] <= highest)]


def main():
    # print("Test 1: Read a trackfile, present parameters")
    # before = read_trackfile("for003_G4BL_JINST_120_1p4_1p0.dat")
    # print_all_params(before)

    # print("Test 2: Generate trackfile, verify that parameters are correct")
    # generated = gen_distribution(
    #     (0.017, 0.7, 0.115, 0, 0),
    #     (0.017, 0.7, 0.115, 0, 0),
    #     120,
    #     1,
    #     z_emit=1,
    # )
    # print_all_params(generated)
    # print(len(generated))
    # write_trackfile(generated, filepath="particles_before.txt", comment="particles_before")

    # print("Test 3: Plot histograms")
    # def plot_hist(df, x_field, y_field, label=None):
    #     fig, ax = plt.subplots()
    #     ax.hist2d(
    #         df[x_field],
    #         df[y_field],
    #         bins=(
    #             np.linspace(np.min(df[x_field]), np.max(df[x_field]), 50),
    #             np.linspace(np.min(df[y_field]), np.max(df[y_field]), 50),
    #         ),
    #     )
    #     ax.set_xlabel(x_field)
    #     ax.set_ylabel(y_field)
    #     if label:
    #         fig.suptitle(label)
    #
    # plot_hist(before, "x", "y", label="Before")
    # plot_hist(before, "x", "Px", label="Before")
    # plot_hist(before, "y", "Py", label="Before")
    # plt.show()

    # print("Test 4: Generate a distribution, run G4Beamline")
    # generated = gen_distribution(
    #     (0.0272, 0.407, 0.05, 0.001119, -0.00161),
    #     (0.0270, 0.392, 0.05, -0.0001, 0.03157),
    #     30,
    #     10,
    #     1.1875,
    # )
    # result = run_distribution(generated, length=7, half_angle=45)
    # print(result)
    # print("Statistics before:")
    # print_all_params(generated)
    # print()
    # print("Statistics after:")
    # print_all_params(result)

    # print("Test 5: Sweep over momentum")
    # results = list()
    # for m in trange(30, 151, 10):
    #     for _ in range(5):
    #         generated = gen_distribution(
    #             (0.0272, 0.407, 0.05, 0.001119, -0.00161),
    #             (0.0270, 0.392, 0.05, -0.0001, 0.03157),
    #             m,
    #             10,
    #             1.1875,
    #         )
    #         result = run_distribution(generated)
    #         results.append([m]+list(emittances(result)))
    # df = pandas.DataFrame(results, columns=["momentum", "xemit", "yemit", "zemit"])
    # print(df)
    # df.to_pickle("momentum_sweep_results.pkl")

    # print("Test 5.1: Read the data we produced in the previous test")
    # df = pd.read_pickle("momentum_sweep_results.pkl")
    # print(df)
    # plt.scatter(df["momentum"], df["xemit"], label="xemit")
    # plt.scatter(df["momentum"], df["yemit"], label="yemit")
    # # plt.scatter(df["momentum"], df["zemit"], label="zemit")
    # plt.legend()
    # plt.show()
    #
    # print(max_length(10), max_length(60))
    #
    # print("Test 6: Scan over angle")
    # results = list()
    # for angle in trange(10, 61, 5):
    #     length = max_length(angle) + 0.1  # give it some wiggle room
    #     for _ in range(5):
    #         generated = gen_distribution(
    #             (0.0272, 0.407, 0.05, 0.001119, -0.00161),
    #             (0.0270, 0.392, 0.05, -0.0001, 0.03157),
    #             120,
    #             10,
    #             1.1875,
    #         )
    #         result = run_distribution(generated, abshalfangle3=angle, absLEN3=length)
    #         results.append([angle, length] + list(emittances(result)))
    # df = pandas.DataFrame(results, columns=["halfangle", "length", "xemit", "yemit", "zemit"])
    # print(df)
    # df.to_pickle("angle_sweep_results.pkl")

    # print("Test 6.1: Read the data we produced in the previous test")
    # df = pd.read_pickle("angle_sweep_results.pkl")
    # print(df)
    # plt.scatter(df["halfangle"], df["xemit"], label="xemit")
    # plt.scatter(df["halfangle"], df["yemit"], label="yemit")
    # # plt.scatter(df["halfangle"], df["zemit"], label="zemit")
    # plt.legend()
    # plt.show()
    # results = list()

    # print("Test 7: New distribution function (yes this means all previous tests no longer work)")
    # b = .027
    # a = 0.4
    # before = gen_distribution((b, a, 0.145, 0, 0), (b, a, 0.145, 0, 0), 120, 1, z_emit=1)
    # print(emittances(run_distribution(before, 3, 45)))
    # print(emittances(run_distribution(before, 6, 45)))
    # print(emittances(run_distribution(before, 12, 45)))

    # Some variables for future tests
    # BETA = .03  # centimeter to meter conversion!
    # ALPHA = 0
    # T_EMIT = 0.145
    # L_EMIT = 1

    # print("Test 7.1: Generate up a test case")
    # before = gen_distribution((BETA, ALPHA, T_EMIT, 0, 0), (BETA, ALPHA, T_EMIT, 0, 0), 180, 1, z_emit=L_EMIT)
    # print_all_params(run_distribution(before, 12, 45))

    # print("Test 8: Cuts")
    # before = read_trackfile("particles_after.txt")
    # print(before)
    # after = cut_outliers(before)
    # print(after)

    # print("Test 9: Base length setting working?")
    # before = gen_distribution((BETA, ALPHA, T_EMIT, 0, 0), (BETA, ALPHA, T_EMIT, 0, 0), 120, 1, z_emit=L_EMIT)
    # run_distribution(before, 24, 5, 45, True)

    # print("Test 10: Wait, is generation borked?")
    # before = gen_distribution((BETA, ALPHA, T_EMIT, 0, 0), (BETA, ALPHA, T_EMIT, 0, 0), 120, 1, z_emit=L_EMIT)
    # print_all_params(before)

    # print("Test 11: Removing dispersion")
    # before = gen_distribution((BETA, ALPHA, T_EMIT, 0.02, -0.04), (BETA, ALPHA, T_EMIT, 0, 0), 120, 1, z_emit=L_EMIT)
    # print_all_params(before)
    # after = remove_dispersion(before)
    # print_all_params(after)

    # print("Test 12: Simulating drift")
    # post_wedge = read_trackfile("results/distributions/145_After.txt")
    # post_correct = remove_dispersion(post_wedge)
    # post_drift = z_prop(post_correct, 5000)
    # print(post_drift["t"])

    pass


if __name__ == "__main__":
    main()
