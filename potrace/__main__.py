import argparse
from pathlib import Path

from potrace import potrace
from potrace.common import TurnPolicy

parser = argparse.ArgumentParser(description='Potrace')
parser.add_argument('filename', type=Path, nargs='+', help='an input file')
parser.add_argument('--output', '-o', type=Path, nargs='?', help='write all output to this file')

parser.add_argument('--backend', '-b', choices=["svg"], default="svg", help="select backend by name")
parser.add_argument('--turnpolicy', '-z', type=TurnPolicy.argparse, choices=list(TurnPolicy),
                    help="how to resolve ambiguities in path decomposition", default=TurnPolicy.MINORITY)
parser.add_argument('--turdsize', '-t', type=int, help="suppress speckles of up to this size (default 2)", default=2)
parser.add_argument('--alphamax', '-a', type=float, help="corner threshold parameter (default 1)", default=1.)
parser.add_argument('--longcurve', '-n', type=bool, help="turn off curve optimization", default=True)
parser.add_argument('--opttolerance', '-O', type=float, help="curve optimization tolerance (default 0.2)", default=.2)

if __name__ == "__main__":
    from skimage.io import imread

    args = parser.parse_args()
    for filename in args.filename:
        potrace.potrace(imread(filename),
                        output=str(args.output if args.output is not None else filename) + "." + args.backend,
                        turdsize=args.turdsize,
                        turnpolicy=args.turnpolicy, alphamax=args.alphamax, optcurve=args.longcurve,
                        opttolerance=args.opttolerance)
