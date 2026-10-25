// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! Input events a frontend queues in `KlartraumEngine::event_queue()`; `step()` passes them to the camera.

/// A mouse button.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MouseButton {
    Left,
    Right,
    Middle,
}

/// Whether a button or key went down or up.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    Press,
    Release,
}

/// The keys a frontend reports; others arrive as `Key::Unknown`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[rustfmt::skip]
pub enum Key {
    Space, Escape, Enter, Tab, Backspace, Insert, Delete,
    Right, Left, Down, Up, PageUp, PageDown, Home, End,
    Num0, Num1, Num2, Num3, Num4, Num5, Num6, Num7, Num8, Num9,
    A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z,
    F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12,
    LeftShift, LeftControl, LeftAlt, LeftSuper, RightShift, RightControl, RightAlt, RightSuper,
    Unknown,
}

/// An input event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event {
    /// The cursor moved: position (`x`, `y`) and movement (`dx`, `dy`) since the last event, in screen coordinates.
    MouseMove { x: i32, y: i32, dx: i32, dy: i32 },
    /// A mouse button was pressed or released.
    MouseButton { button: MouseButton, action: Action },
    /// The mouse wheel or touchpad scrolled; (`x`, `y`) is the offset since the last frame.
    MouseScroll { x: f64, y: f64 },
    /// A key was pressed or released.
    Key { key: Key, action: Action },
}
