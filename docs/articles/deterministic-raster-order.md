# Deterministic raster order

```{note}
This is a slop-free technology writeup. Klartraum contains a lot of code and documentation
that has been written by AI. These articles here, however, are hand-written
(apart from a little help with syntax, grammar and style).
```

Gaussian Splatting is implemented by two render backends in Klartraum: one using compute shaders
and one using the normal GPU rasterization pipeline. The former is based on the rendering approach in the original
Gaussian Splatting paper {cite:p}`kerbl2023_3Dgaussians`. In Klartraum, this basically bins the projected Gaussians into a 4x4 grid and then uses a radix sort to sort them by bin and depth before walking through each bin's sorted list for each pixel.
The binning and the radix sort are stable, making the resulting order deterministic, too.

The raster backend culls the Gaussians, sorts the surviving indices by depth and renders a quad per visible Gaussian with
an indirect draw. Rasterization and blending of the quads is in the back-to-front order created
by the radix sort. For equal depth keys, the stable sort preserves the order of its input.
For this algorithm to be deterministic too, the culling step must
create the same list in the same order for the same inputs. A naive implementation of it on the GPU might not preserve that order
and can thus produce unstable rendering results.

## Naive culling

The naive approach for culling uses one thread per Gaussian. It determines for each Gaussian
if it is within the visible bounds and returns early if it isn't. If it is, an offset is
computed with `atomicAdd`. Then, it writes its index in the source array and its depth value
into the sorting arrays for indices and keys at its assigned offset.
Simple, effective, but the order is dependent on thread and workgroup execution order, which is
not deterministic.

## Stable culling

The stable culling replaces this simple approach with three compute passes:
Count, Scan and Scatter.
The first pass counts the number of visible Gaussians per workgroup.
The second pass does a basic Hillis-Steele scan {cite:p}`harris2007_parallelScan` over the counts from the previous step and converts the result to exclusive offsets (i.e., totals for all earlier workgroups).
This gives the starting offset for each workgroup in the output arrays,
which is used in the last step.
In the last step, again Hillis-Steele is used on the visibility of the Gaussians within each workgroup,
giving each visible Gaussian its rank within the workgroup, starting at one. Subtracting one and adding the workgroup's offset gives its output location.
This is then finally used to scatter
the Gaussian index and depth value to the sorting arrays.

The result of each step does not depend on thread and workgroup execution order, making the procedure deterministic.

## Runtime costs

The stability achieved by the algorithm is not for free. For the lantern example, it has been measured
that the culling step runtime increased from 0.08 ms to 0.12 ms (Apple M4 with MoltenVK).
The extra 0.04 ms amounts to only about 0.7 % of the 5.8 ms spent in the raster stages and is acceptable.

## References

```{bibliography}
:filter: docname in docnames
```
