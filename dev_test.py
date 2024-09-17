import argparse


parser = argparse.ArgumentParser(description="Test the development environment")
parser.add_argument("--grad_modif_params", type=str, nargs="+", default=[], required=False)

args = parser.parse_args()
print(args.grad_modif_params)