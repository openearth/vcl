"""Console script for vcl."""
import concurrent.futures
import sys

import click
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, mgrid, pi, sin


def mayavi_window():
    import mayavi
    from mayavi import mlab

    # Create the data.
    dphi, dtheta = pi/250.0, pi/250.0
    [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
    m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
    r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
    x = r*sin(phi)*cos(theta)
    y = r*cos(phi)
    z = r*sin(phi)*sin(theta)

    # View it.
    s = mlab.mesh(x, y, z)
    mlab.show()


def opencv_window():
    img = np.zeros([100, 100, 3])
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.imshow("window", img)
    while True:
        k = cv2.waitKey(0)
        if k == ord('f'):
            cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        elif k == ord('n'):
            cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif k == ord('q'):
            cv2.destroyWindow("window")
            break
        else:
            break



def matplotlib_window():
    matplotlib.use('qtagg')
    fig, ax = plt.subplots()
    ax.plot([1, 2], [1, 2])
    # keep window open
    plt.show()



@click.command()
def main(args=None):
    """Console script for vcl."""

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)
    executor.submit(matplotlib_window)
    # executor.submit(mayavi_window)
    executor.submit(opencv_window)

    # return exit status 0
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
