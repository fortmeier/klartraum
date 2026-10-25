// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

//! [`WindowFrontend`], a windowed application with a render loop (winit; the C++ engine uses GLFW).

use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowId};

use crate::engine::KlartraumEngine;
use crate::events::{Action, Event, Key, MouseButton};
use crate::vulkan_context::{BackendConfig, Presentation, VulkanContext};

/// The window and the input state gathered from its events between frames.
struct App {
    window: Option<Window>,
    /// An error from creating the window, reported by `WindowFrontend::new`.
    window_error: Option<String>,
    close_requested: bool,
    /// The latest cursor position in logical pixels, and the one reported to the engine last.
    cursor: (i32, i32),
    reported_cursor: (i32, i32),
    left_button: Option<Action>,
    scroll: (f64, f64),
    keys: Vec<(Key, Action)>,
    /// The latest framebuffer size, if it changed since the last frame.
    resized: Option<PhysicalSize<u32>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attributes = Window::default_attributes()
            .with_title(BackendConfig::ENGINE_VERSION)
            .with_inner_size(PhysicalSize::new(BackendConfig::WIDTH, BackendConfig::HEIGHT))
            .with_resizable(false);
        match event_loop.create_window(attributes) {
            Ok(window) => self.window = Some(window),
            Err(e) => self.window_error = Some(e.to_string()),
        }
    }

    fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let scale = self.window.as_ref().map_or(1.0, Window::scale_factor);
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            WindowEvent::Resized(size) => self.resized = Some(size),
            WindowEvent::CursorMoved { position, .. } => {
                let logical = position.to_logical::<f64>(scale);
                self.cursor = (logical.x as i32, logical.y as i32);
            }
            WindowEvent::MouseInput {
                state,
                button: winit::event::MouseButton::Left,
                ..
            } => {
                self.left_button = Some(action(state));
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let (x, y) = match delta {
                    MouseScrollDelta::LineDelta(x, y) => (x as f64, y as f64),
                    // Trackpads report pixels; GLFW scales them by 0.1 on macOS.
                    MouseScrollDelta::PixelDelta(p) => {
                        let logical = p.to_logical::<f64>(scale);
                        (logical.x * 0.1, logical.y * 0.1)
                    }
                };
                self.scroll.0 += x;
                self.scroll.1 += y;
            }
            WindowEvent::KeyboardInput { event, .. } if !event.repeat => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.keys.push((translate_key(code), action(event.state)));
                }
            }
            _ => {}
        }
    }
}

fn action(state: ElementState) -> Action {
    match state {
        ElementState::Pressed => Action::Press,
        ElementState::Released => Action::Release,
    }
}

fn translate_key(code: KeyCode) -> Key {
    use KeyCode as C;
    match code {
        C::Space => Key::Space,
        C::Escape => Key::Escape,
        C::Enter => Key::Enter,
        C::Tab => Key::Tab,
        C::Backspace => Key::Backspace,
        C::Insert => Key::Insert,
        C::Delete => Key::Delete,
        C::ArrowRight => Key::Right,
        C::ArrowLeft => Key::Left,
        C::ArrowDown => Key::Down,
        C::ArrowUp => Key::Up,
        C::PageUp => Key::PageUp,
        C::PageDown => Key::PageDown,
        C::Home => Key::Home,
        C::End => Key::End,
        C::Digit0 => Key::Num0,
        C::Digit1 => Key::Num1,
        C::Digit2 => Key::Num2,
        C::Digit3 => Key::Num3,
        C::Digit4 => Key::Num4,
        C::Digit5 => Key::Num5,
        C::Digit6 => Key::Num6,
        C::Digit7 => Key::Num7,
        C::Digit8 => Key::Num8,
        C::Digit9 => Key::Num9,
        C::KeyA => Key::A,
        C::KeyB => Key::B,
        C::KeyC => Key::C,
        C::KeyD => Key::D,
        C::KeyE => Key::E,
        C::KeyF => Key::F,
        C::KeyG => Key::G,
        C::KeyH => Key::H,
        C::KeyI => Key::I,
        C::KeyJ => Key::J,
        C::KeyK => Key::K,
        C::KeyL => Key::L,
        C::KeyM => Key::M,
        C::KeyN => Key::N,
        C::KeyO => Key::O,
        C::KeyP => Key::P,
        C::KeyQ => Key::Q,
        C::KeyR => Key::R,
        C::KeyS => Key::S,
        C::KeyT => Key::T,
        C::KeyU => Key::U,
        C::KeyV => Key::V,
        C::KeyW => Key::W,
        C::KeyX => Key::X,
        C::KeyY => Key::Y,
        C::KeyZ => Key::Z,
        C::F1 => Key::F1,
        C::F2 => Key::F2,
        C::F3 => Key::F3,
        C::F4 => Key::F4,
        C::F5 => Key::F5,
        C::F6 => Key::F6,
        C::F7 => Key::F7,
        C::F8 => Key::F8,
        C::F9 => Key::F9,
        C::F10 => Key::F10,
        C::F11 => Key::F11,
        C::F12 => Key::F12,
        C::ShiftLeft => Key::LeftShift,
        C::ControlLeft => Key::LeftControl,
        C::AltLeft => Key::LeftAlt,
        C::SuperLeft => Key::LeftSuper,
        C::ShiftRight => Key::RightShift,
        C::ControlRight => Key::RightControl,
        C::AltRight => Key::RightAlt,
        C::SuperRight => Key::RightSuper,
        _ => Key::Unknown,
    }
}

/// A window of [`BackendConfig::WIDTH`] x [`BackendConfig::HEIGHT`] pixels with an
/// engine rendering into its swapchain, the mouse and keyboard driving the camera.
pub struct WindowFrontend {
    /// Declared first, so the engine (and the surface) is dropped before the window.
    engine: KlartraumEngine,
    app: App,
    event_loop: EventLoop<()>,
}

impl WindowFrontend {
    pub fn new() -> Result<Self> {
        let mut event_loop = EventLoop::new().context("failed to create the event loop")?;
        let mut app = App {
            window: None,
            window_error: None,
            close_requested: false,
            cursor: (0, 0),
            reported_cursor: (0, 0),
            left_button: None,
            scroll: (0.0, 0.0),
            keys: Vec::new(),
            resized: None,
        };
        // The window is created when the event loop resumes the application.
        while app.window.is_none() {
            if let Some(e) = app.window_error.take() {
                return Err(anyhow!("failed to create the window: {e}"));
            }
            if let PumpStatus::Exit(code) = event_loop.pump_app_events(Some(Duration::ZERO), &mut app) {
                return Err(anyhow!("the event loop exited ({code}) before a window was created"));
            }
        }
        let window = app.window.as_ref().unwrap();
        let size = window.inner_size();
        let presentation = Presentation::Window {
            display: window.display_handle()?.as_raw(),
            window: window.window_handle()?.as_raw(),
            framebuffer_extent: vk::Extent2D {
                width: size.width,
                height: size.height,
            },
        };
        let engine = KlartraumEngine::new(VulkanContext::new(presentation)?);
        Ok(Self {
            engine,
            app,
            event_loop,
        })
    }

    pub fn engine(&mut self) -> &mut KlartraumEngine {
        &mut self.engine
    }

    pub fn window(&self) -> &Window {
        self.app.window.as_ref().unwrap()
    }

    fn pump(&mut self, timeout: Option<Duration>) {
        if let PumpStatus::Exit(_) = self.event_loop.pump_app_events(timeout, &mut self.app) {
            self.app.close_requested = true;
        }
        if let Some(size) = self.app.resized.take() {
            self.engine.vulkan_context_mut().set_framebuffer_extent(vk::Extent2D {
                width: size.width,
                height: size.height,
            });
        }
    }

    /// Queues the input gathered since the last frame as engine events.
    fn queue_events(&mut self) {
        let app = &mut self.app;
        let queue = self.engine.event_queue();
        if app.cursor != app.reported_cursor {
            let (x, y) = app.cursor;
            queue.push_back(Event::MouseMove {
                x,
                y,
                dx: x - app.reported_cursor.0,
                dy: y - app.reported_cursor.1,
            });
            app.reported_cursor = app.cursor;
        }
        if let Some(action) = app.left_button.take() {
            queue.push_back(Event::MouseButton {
                button: MouseButton::Left,
                action,
            });
        }
        if app.scroll != (0.0, 0.0) {
            queue.push_back(Event::MouseScroll {
                x: app.scroll.0,
                y: app.scroll.1,
            });
            app.scroll = (0.0, 0.0);
        }
        for (key, action) in app.keys.drain(..) {
            queue.push_back(Event::Key { key, action });
        }
    }

    /// Renders frames until the window is closed, or after `max_frames` frames if it is positive.
    pub fn run(&mut self, max_frames: i64) -> Result<()> {
        // Resizing is offered only when the engine can rebuild its graphs for a new swapchain.
        self.window().set_resizable(self.engine.is_resizable());
        let mut frames = 0;
        while !self.app.close_requested {
            self.pump(Some(Duration::ZERO));
            // A minimized window has nothing to present to; wait until it is restored.
            while !self.app.close_requested && {
                let size = self.window().inner_size();
                size.width == 0 || size.height == 0
            } {
                self.pump(None);
            }
            if self.app.close_requested {
                break;
            }
            self.queue_events();
            self.engine.step()?;
            frames += 1;
            if max_frames > 0 && frames >= max_frames {
                break;
            }
        }
        self.engine.vulkan_context().stop_render();
        Ok(())
    }
}
