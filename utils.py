import argparse
import gzip
import os
import _pickle as pickle


def pickle_path():
    pickle_path = os.path.join('.', 'pickles')
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
    return pickle_path


def pickle_save(outfile, res):
    outpath = os.path.join(pickle_path(), outfile)
    pickle.dump(res, gzip.open(outpath, 'wb'))
    print("Results saved to " + outpath)


def plot_path():
    plot_path = os.path.join('.', 'plots')
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    return plot_path


def plot_save(fig, outfile):
    outpath = os.path.join(plot_path(), outfile)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    print("Plot saved to " + outpath)


def fetch_dim_args(lat_default):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat_x", type=int, help="x-length of lattice")
    parser.add_argument("--lat_y", type=int, help="y-length of lattice")
    parser.add_argument("--grid_x", type=int, help="x-length of process grid")
    parser.add_argument("--grid_y", type=int, help="y-length of process grid")
    args = parser.parse_args()
    
    lat_dims = [
            args.lat_x, args.lat_y
        ] if args.lat_x is not None and args.lat_y is not None else lat_default
    
    grid_dims = [
            args.grid_x, args.grid_y
        ] if args.grid_x is not None and args.grid_y is not None else None

    return lat_dims, grid_dims