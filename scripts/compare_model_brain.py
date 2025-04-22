from PIL import Image
import re
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.io as sio  # load .mat file
import sys
import argparse
import os
import glob
import os.path as osp
import time
import matplotlib.pyplot as plt


N_SAMPLES = 1000
COLUMNS = ["Region", "ReferenceRegion", "r"]
NETWORK_LAYERS = ["FC_MU", "FC_VAR", "Z", "DEC_FC", "TCL", "POOL"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", default="*[!Brain]",
                        help="""Input regions: Specify directories which to recursively search .pkl files""")
    parser.add_argument("--ref_dir", default="Brain/rdms",
                        help="""Reference region: Specify directory in which to recursively search .pkl file""")
    parser.add_argument("--ref_regions", default=["MSB", "ASB"], nargs="+",
                        help="""Specify reference regions""")
    parser.add_argument("--out_name", default="model-v-brain",
                        help="""Name of the output .pkl file""")
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--holistic", action="store_true")
    return parser.parse_args()

def rdm_similarity(rdm1, rdm2, n_samples=100):
    assert rdm1.shape == rdm2.shape
    n = rdm1.shape[0]
    ut = np.triu_indices(n, 1)

    corrs = np.zeros(n_samples)
    # use the same stimulus pairs for all RSM comparisons
    np.random.seed(0)
    for idx in range(n_samples):
        # sample stimulus pairs with replacement
        sample = np.random.choice(n, n)
        m = rdm2[sample, :][:, sample]
        d = rdm1[sample, :][:, sample]

        # take the upper triangle (excluding the diagonal)
        m = m[ut]
        d = d[ut]

        # exclude correlations that are exactly 1 (artefacts of the diagonal being sampled)
        m_final = m[m != 1]
        d = d[m != 1]

        # determine rank correlation
        r, _ = stats.spearmanr(m_final, d)
        corrs[idx] = r

    return corrs

def compare_rdms(rdm_df_comp, rdm_df_ref, out_pth, ref_regions=None, holistic=False, **kwargs):
    all_data = []
    best_data = {}
    # HACK change "+" to "&" in ref_regions
    if not holistic:
        ref_regions = [i.replace("+", "&") for i in ref_regions]
    best_rdm_comps = []
    for ref_region in rdm_df_ref.Region.unique():
        if ref_regions is not None and ref_region not in ref_regions:
            continue
            # extract rdm for current reference region
        rdm_ref = rdm_df_ref[rdm_df_ref.Region == ref_region].RDM.values[0]
        best_rs = np.zeros(1) # to keep track of best correlation
        for region in rdm_df_comp.Region.unique():
            # extract rdm for current region to compare with reference region
            rdm_comp = rdm_df_comp[rdm_df_comp.Region == region].RDM.values[0]
            rs = rdm_similarity(rdm_comp, rdm_ref, **kwargs)
            # r_mean = np.mean(rs)
            # res = stats.bootstrap((rs,), np.mean).confidence_interval
            # r_min, r_max = res.low, res.high
            # all_data.append([region, ref_region, r_mean, r_min, r_max])
            all_data.append([region, ref_region, rs])
            # if the layer is a network layer, record it
            region_split = region.split("_")
            is_network = region in NETWORK_LAYERS or len(region_split) == 1 and bool(re.search(r'\d', region_split[0]))
            if rs.mean() > best_rs.mean() and is_network:
                best_region = region
                best_rs = rs
                best_rdm_comp = rdm_comp
        best_data[f"best_{ref_region}_r"] = best_rs
        best_data[f"best_{ref_region}_region"] = best_region
        best_rdm_comps.append(best_rdm_comp)

    df = pd.DataFrame(all_data, columns=COLUMNS)
    info_df = rdm_df_comp.drop(columns=["RDM", "VariableOrder"])
    df = df.merge(info_df.drop_duplicates(), how="inner", on="Region")
    df = df.explode("r").reset_index()
    df.to_pickle(out_pth)
    print(f"Saved to {out_pth}")

    # correlations within best rdms per reference regions
    if ref_regions == ["MSB", "ASB"]:
        rs = rdm_similarity(*best_rdm_comps, **kwargs)
        best_data["Within_r"] = rs
        best_data["Name"] = df.Name.unique()[0]
        best_data["Source"] = df.Source.unique()[0]
        best_data["Backbone"] = df.Backbone.unique()[0]
        best_data["Composite"] = best_data["best_MSB_r"] + best_data["best_ASB_r"] - best_data["Within_r"]
        best_pth = osp.join(osp.dirname(out_pth), "IT_score.pkl")
        df = pd.DataFrame.from_dict(best_data)
        df.to_pickle(best_pth)
        print(f"Saved to {best_pth}")


def main():
    args = parse_args()
    networks_dir = "data/networks"
    suffix = "" if not args.holistic else "_holistic"
    #HACK
    args.ref_regions = ["MSB+ASB"] if args.holistic else args.ref_regions
    IN_FILE_NAME = f"rdms{suffix}.pkl"
    out_folder = "comparisons"
    print("Comparing to regions:", args.ref_regions)

    # gather reference rdms (e.g. brain)
    ref_rdms = pd.read_pickle(osp.join(networks_dir, args.ref_dir, IN_FILE_NAME))

    # gather input rdms (e.g. models)
    search_dir = osp.join(networks_dir, args.in_dir, "**", IN_FILE_NAME)
    rdm_pths = glob.glob(search_dir, recursive=True)

    for rdm_pth in rdm_pths:
        print(f"Working on {rdm_pth}")
        in_rdms = pd.read_pickle(rdm_pth)

        out_pth = osp.join(osp.dirname(osp.dirname(rdm_pth)), out_folder, args.out_name+suffix+".pkl")
        if osp.exists(out_pth) and args.skip: continue
        os.makedirs(osp.dirname(out_pth), exist_ok=True)
        compare_rdms(in_rdms, ref_rdms, out_pth, ref_regions=args.ref_regions,
                     n_samples=N_SAMPLES, holistic=args.holistic)

if __name__ == "__main__":
    main()
