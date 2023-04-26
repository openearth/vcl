"""Console script for vcl."""
import sys
import click

import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


@click.command()
def main(args=None):
    """Console script for vcl."""

    # empty image
    img = np.zeros([100, 100, 3])
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.imshow("window", img)
    cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    matplotlib.use('qtagg')
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    # keep window open

    plt.show()



    # return exit status 0
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
