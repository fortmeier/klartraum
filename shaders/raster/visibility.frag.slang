// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Writes the visible triangle's index plus one; the target is cleared to zero,
// which marks pixels no triangle covers.
import visibility_common;

[shader("fragment")]
uint main(VisibilityVertex input) : SV_Target0 {
    return input.triangle + 1;
}
