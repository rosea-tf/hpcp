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

def fetch_grid_dims():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_x", type=int, help="x-length of process grid")
    parser.add_argument("--grid_y", type=int, help="y-length of process grid")
    args = parser.parse_args()
    
    try:
        grid_dims = [
            args.grid_x, args.grid_y
        ] if args.grid_x is not None and args.grid_y is not None else None
    except:
        grid_dims = None
    
    return grid_dims