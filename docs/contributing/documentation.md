# Writing documentation

## Building the documentation

The documentation is built with [Sphinx](https://www.sphinx-doc.org/) and
[MyST Markdown](https://myst-parser.readthedocs.io/). The C++ API reference is
extracted by [Doxygen](https://www.doxygen.nl/) and rendered by
[Breathe](https://breathe.readthedocs.io/). You need Python 3 and Doxygen.

```bash
python3 -m venv docs/.venv
docs/.venv/bin/pip install -r docs/requirements.txt
docs/.venv/bin/sphinx-build -W --keep-going docs docs/_build/html
```

`conf.py` runs Doxygen itself, so no separate step is needed. Open
`docs/_build/html/index.html` to view the result. `-W` turns warnings into
errors, as in CI. Doxygen warnings fail the build too, for example an `@param`
that names no parameter of the function; they are listed in
`docs/_doxygen/warnings.log`.

## Citing sources in articles

Add BibTeX entries to the shared `docs/references.bib` file. Cite their keys in
MyST Markdown with `{cite:t}` for a citation in the sentence or `{cite:p}` for
a parenthetical citation:

```markdown
The original method is described by {cite:t}`kerbl2023_3Dgaussians`.
The renderer uses Gaussian splats {cite:p}`kerbl2023_3Dgaussians`.
```

End each article that cites sources with:

````markdown
## References

```{bibliography}
:filter: docname in docnames
```
````

The filter includes only sources cited on that page. Several articles can cite
the same entry; each citation links to the bibliography on its own page.

## Documenting C++ code

Document the public API in the headers under `include/klartraum/`, using
Doxygen comments:

```cpp
/**
 * @brief Concatenates two sets of Gaussians every time the graph runs.
 *
 * A's Gaussians come first, so one backend renders and depth-sorts both.
 * The output is per-path.
 *
 * @see GaussianTransform
 */
class GaussianMerge : public GeneralComputation<GaussianMergePushConstants> {
public:
    /**
     * @brief Creates the merge element.
     * @param a First input; its Gaussians come first in the output.
     * @param b Second input, appended after @p a.
     */
    GaussianMerge(GaussianDataPtr a, GaussianDataPtr b);

    uint32_t count;  ///< Number of Gaussians in the merged output.
};
```

- Use `/** ... */` blocks with `@brief`, `@param`, `@return`, `@throws`,
  `@note` and `@see`, and `///<` for short comments after members.
- Markdown (lists, `code`, emphasis) works inside the comments.
- Comments in `.cpp` files and on private members stay ordinary `//`
  comments; they are not part of the API reference.
- Longer explanations of concepts belong in these documentation pages, not
  in header comments.

## Adding a class to the API reference

The API pages in `docs/api/` list classes explicitly:

````md
```{doxygenclass} klartraum::GaussianMerge
```
````

Prose pages can link to documented classes with
``{cpp:class}`klartraum::GaussianMerge` ``.
