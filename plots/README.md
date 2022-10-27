Some example plots
====
Using some git trickery, I've put the plots generated in the latest CI below.

Mixing Toy
----
D mixing arises because of interference between $D^0$ and $\overline{D}^0$ amplitudes.
This interference can be quantified, e.g.:

$D^0(t) = g_+(t)D^0 + \frac{q}{p}g_-(t)\overline{D}^0$

$g_-(t) = \mathrm{e}^{-iMt-\frac{1}{2}\Gamma_D t} \mathrm{cos}\left(\frac{1}{2}\Delta M t - \frac{i}{4} \Delta \Gamma t\right)$

We can plot the magnitude of the coefficient $g_-(t)$ to tell us how the magnitude of the $D^0$ and $\overline{D}^0$
components evolve with time:
![mixing](/../example_plots/mixing.png)

Clearly the time dependence is dominated by the exponential decay of the $D^0$ - i.e., charm mixing is a small effect.

We can take the log to account for this...

![log mixing](/../example_plots/log_mixing.png)

There's still hardly any mixing - we can turn the mixing effect up by increasing the value of the $D^0$ mass difference
by a lot:

![more mixing](/../example_plots/more_mixing.png)

We can now clearly see the D oscillating between the $D^0$ and $\overline{D}^0$ states.

Mass Fit Toy
----
With some toy $\Delta M$ values (generated from the fitting PDF), we can do a mass fit:

![mass fit](/../example_plots/toy_mass_fit.png)

Time Fitter Toys
----
We can similarly generate some toy decay times with some mixing and fit to them.

In this case we take our expected decay rates (exponential for RS, exponential times a polynomial for WS)
and generate some decay times from them.

We then take the ratio and can perform fits:

![fit examples](/../example_plots/fit_examples.png)

We can also do a scan over a fit like this - this is how we extract $Z_\Omega^f$:

![scan](/../example_plots/scan.png)

We can do many of these scans and see how often the true value is within our 1$\sigma$ band:

![coverage](/../example_plots/coverage.png)

The Gaussian constraint on $x$ and $y$ looks like this:

![constraint](/../example_plots/xy_constraint.png)

