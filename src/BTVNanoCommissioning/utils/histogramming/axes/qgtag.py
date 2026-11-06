import hist
import numpy as np

from .common import axes as common_axes

axes = common_axes | {
    "flav": hist.axis.IntCategory([0, 1, 4, 5, 6], name="flav", label="Genflavour"),
    # PartonFlavour categories: 0=ud, 1=s, 2=c, 3=b, 4=g, 5=other
    "pflav": hist.axis.IntCategory([0, 1, 2, 3, 4, 5], name="flav", label="Genflavour"),
    "pt": hist.axis.Variable(
        np.array([20, 30, 50, 70, 100, 140, 200, 300, 600, 1000, 2000, 4000, 6800]),
        name="pt",
        label=" $p_{T}$ [GeV]",
    ),
    "eta": hist.axis.Variable(
        [-5.131, -4.7, -3.0, -2.5, -1.3, 0.0, 1.3, 2.5, 3.0, 4.7, 5.131],
        name="eta",
        label=" $\eta$",
    ),
}
