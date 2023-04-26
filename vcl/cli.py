"""Console script for vcl."""
import sys
import click

import matplotlib
import matplotlib.pyplot as plt


@click.command()
def main(args=None):
    """Console script for vcl."""
    matplotlib.use('qtagg')
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    # return exit status 0
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
