# Code style

These rules apply to the C++ sources in `include/`, `src/`, `tests/` and
`examples/`. Most of them are checked before each commit (see
{doc}`development`); the rest is up to review.

## Formatting

clang-format formats every `.cpp` and `.hpp` file according to
`.clang-format`, and the pre-commit check enforces it:

- four spaces, no tabs, at most 120 characters per line
- braces on the same line: `if (x) {`, `class A {`, `void f() {`
- `template <...>` on its own line, constructor initializers one per line
- pointer and reference symbols next to the type: `int* p`, `const T& v`
- no automatic alignment of `=` or declarations

Includes are not sorted automatically yet. Keep the existing order: standard
library, other libraries, then project headers.

All text files use UTF-8, LF line endings, no trailing whitespace and end with
a newline (`.editorconfig`, checked before each commit).

## Headers

Every header has an include guard named after its path; `#pragma once` is not
used. The check before each commit enforces it, and
`python3 scripts/dev/include_guards.py --fix` adds or renames guards.

```cpp
// include/klartraum/computegraph/copybuffer.hpp
#ifndef KLARTRAUM_COMPUTEGRAPH_COPYBUFFER_HPP
#define KLARTRAUM_COMPUTEGRAPH_COPYBUFFER_HPP
...
#endif // KLARTRAUM_COMPUTEGRAPH_COPYBUFFER_HPP
```

Public headers live in `include/klartraum/` and are included as
`"klartraum/..."`. All code is in the namespace `klartraum`.

## Naming

| What | Style | Example |
|---|---|---|
| Classes, structs, enums | PascalCase | `ComputeGraph`, `GsplatBackend` |
| Functions and methods | camelCase | `compileFrom()`, `submitAndWait()` |
| Accessors | `get` + name; `is`/`has` for booleans | `getVulkanContext()`, `isResizable()` |
| Compute graph hooks | `_` + camelCase | `_setup()`, `_record()`, `_update()` |
| Variables, parameters, members | camelCase, no prefix or suffix | `numberPaths`, `vulkanContext` |
| Constants | `k` + PascalCase | `kLanternPath` |
| Enum values | PascalCase | `GsplatBackend::Raster` |
| Files | snake_case | `gaussian_splat_rasterizer.hpp` |

Members carry no marker (no `m_`, no trailing `_`). Where a parameter has the
same name as a member, use the initializer list, or `this->` in the body:

```cpp
OffscreenTarget(VulkanContext& vulkanContext, ...) : vulkanContext(vulkanContext) {}

void setOverlay(std::shared_ptr<FrameOverlay> overlay) { this->overlay = std::move(overlay); }
```

When an accessor already has the plain name, give the member a descriptive
name instead (`buffers()` returns `soaBuffers`).

Some existing code does not follow the table yet: a few constants and enum
values are in `ALL_CAPS`, the GLFW callbacks are in snake_case, and the files in
`include/klartraum/computegraph/` join words without underscores
(`bufferelement.hpp`). New code follows the table; existing names are changed
only together with other work on that code.

## Comments

Comments explain what the code does and why, e.g. an invariant or a reason that
is not obvious from the code. When a change fixes a bug, the comment at the
change describes the code as it is now, not the fix or the earlier behaviour;
that history belongs in the commit message.

A `TODO:` says what is missing and why, e.g.
`// TODO: the pool is sized for at most 3 paths; size it by numberPaths.`
Do not commit commented-out code.

## Copyright and license

Every source file starts with the SPDX header; see {doc}`development`.

```cpp
// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT
```

## Tests

- Tests use GoogleTest and live in `tests/test_<topic>.cpp`; add new files to
  `klartraum_tests` in `CMakeLists.txt`.
- Each test file starts (after the SPDX header) with a `TESTS:` block that lists
  every test by name with a one-line summary:

  ```cpp
  /**
   * TESTS:
   * - compilesAndSubmitsAsGraphLeaf: a DrawIndirectCommandBufferElement ...
   * - partialResetPreservesVertexCount: setRecordToZeroRange resets only ...
   **/
  ```

- Order tests from general to specific: basic cases first, real data last.
- Tests that write images write them to `build/TestingOutput/`.
- Tolerances for GPU results come from measurements on the supported drivers,
  with the measured values in a comment.
- Gaussian splatting tests use the lantern scene through `tests/test_scene.hpp`.

## Commits

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):
`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `style:`, `chore:`, `perf:`.
Commits that only reformat code go into `.git-blame-ignore-revs`.
