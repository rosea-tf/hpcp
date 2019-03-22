import argparse

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