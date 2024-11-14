import argparse


def introduce_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--name", type=str)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    return parser
