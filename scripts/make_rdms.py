from PIL import Image
import glob
import pandas as pd
import numpy as np
import scipy.io as sio
import sys
import argparse
import os
import os.path as osp
import time
import matplotlib.pyplot as plt


COLUMNS = ["Region", "RDM", "VariableOrder"]
IN_FILE_NAME = "activations.pkl"
OUT_FILE_NAME = "rdms.pkl"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob_dir", default="*",
                        help="""Specify wildcard for directories
                        in which to recursively search .pkl files""")
    parser.add_argument("--plot", action="store_true", help="Whether to plot the RDMs")
    parser.add_argument("--img-type", default="png", choices=["png", "svg", "pdf"], help="Image type")
    return parser.parse_args()

def build_rdms(activations_df, out_dir, plot, img_type):
    data = []
    var_order = ["View", "Posture", "Identity"]
    # how many values per variables are there?
    line_vars = activations_df[var_order].max().values
    regions = activations_df.Region.unique()
    for region in regions:
        layer_df = activations_df[activations_df.Region == region].copy()
        layer_df.sort_values(by=var_order, inplace=True)
        layer_activations = np.vstack(layer_df.Activations.values)
        rdm = np.corrcoef(layer_activations)
        if plot:
            out_pth = osp.join(out_dir, region+"."+img_type)
            rdm_plot(rdm, region, out_pth, line_vars=line_vars)
        data.append([region, rdm, var_order])
    df = pd.DataFrame(data, columns=COLUMNS)

    # we're collapsing over individual images, so get rid of the respective cols
    info_df = activations_df.drop(
        columns=["Activations", "ImageName", "View", "Posture", "Identity"])

    # merge in the other remaining information from activations_df
    df = df.merge(info_df.drop_duplicates(), how="inner", on="Region")
    return df

def rdm_plot(rsm_array, region, out_pth, line_vars=None, normalize=True):
    fig, ax = plt.subplots()
    offdiag_mask = ~np.eye(rsm_array.shape[0],dtype=bool)
    offdiag_rsm = rsm_array[offdiag_mask]
    if normalize:
        rsm_array[offdiag_mask] = (offdiag_rsm - np.min(offdiag_rsm)) / \
            (np.max(offdiag_rsm) - np.min(offdiag_rsm))
    im = ax.pcolormesh(rsm_array, vmin=0, vmax=1, cmap="Reds", rasterized=True)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.locator_params(nbins=4)
    # draw lines according to the variable
    if line_vars is not None:
        n = rsm_array.shape[0]
        line_dists = n / np.cumprod(line_vars)
        base_lw = 0.4
        for line_level, dist in enumerate(line_dists[:-1]):
            lw = base_lw * (0.6)**line_level
            markers = np.arange(dist, n, dist)
            plt.vlines(markers, 0, n, lw=lw, colors="black")
            plt.hlines(markers, 0, n, lw=lw, colors="black")
    ax.set_aspect("equal")
    ax.set_title(region)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # show only the outside spines
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(out_pth)
    plt.close()

def main():
    args = parse_args()
    networks_dir = "data/networks"
    out_folder = "rdms"

    search_dir = osp.join(networks_dir, args.glob_dir, "**", IN_FILE_NAME)
    activation_files = glob.glob(search_dir, recursive=True)

    for activation_file in activation_files:
        # read the activations
        print(f"Working on {activation_file}")
        activations_df = pd.read_pickle(activation_file)

        out_dir = osp.join(osp.dirname(activation_file), out_folder)
        os.makedirs(out_dir, exist_ok=True)

        df = build_rdms(activations_df, out_dir, plot=args.plot, img_type=args.img_type)

        out_pth = osp.join(out_dir, OUT_FILE_NAME)
        df.to_pickle(out_pth)
        print(f"Saved to {out_pth}")

if __name__ == "__main__":
    main()
