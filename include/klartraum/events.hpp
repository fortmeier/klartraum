#ifndef KLARTRAUM_EVENTS_HPP
#define KLARTRAUM_EVENTS_HPP

namespace klartraum
{
    
class Event
{
public:
    Event() = default;
    virtual ~Event() = default;
};

class EventMouseMove : public Event
{
public:
    int x;
    int y;
    int dx;
    int dy;

    EventMouseMove(int x, int y, int dx, int dy) : x(x), y(y), dx(dx), dy(dy) {}
};

class EventMouseButton : public Event
{
public:
    enum class Button
    {
        Left,
        Right,
        Middle

    } button;

    enum class Action
    {
        Press,
        Release

    } action;

    EventMouseButton(Button button, Action action) : button(button), action(action) {}
};

class EventMouseScroll : public Event
{
public:
    double x;
    double y;
    EventMouseScroll(double x, double y) : x(x), y(y) {}

};

class EventKey : public Event
{
public:
    enum class Key
    {
        Unknown,
        Space,
        Apostrophe,
        Comma,
        Minus,
        Period,
        Slash,
        Num0,
        Num1,
        Num2,
        Num3,
        Num4,
        Num5,
        Num6,
        Num7,
        Num8,
        Num9,
        Semicolon,
        Equal,
        A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,
        LeftBracket,
        Backslash,
        RightBracket,
        GraveAccent,
        World1,
        World2,
        Escape,
        Enter,
        Tab,
        Backspace,
        Insert,
        Delete,
        Right,
        Left,
        Down,
        Up,
        PageUp,
        PageDown,
        Home,
        End,
        CapsLock,
        ScrollLock,
        NumLock,
        PrintScreen,
        Pause,
        F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19,F20,F21,F22,F23,F24,F25,
        KP_0,KP_1,KP_2,KP_3,KP_4,KP_5,KP_6,KP_7,KP_8,KP_9,
        KP_Decimal,KP_Divide,KP_Multiply,KP_Subtract,KP_Add,KP_Enter,KP_Equal,
        LeftShift,LeftControl,LeftAlt,LeftSuper,
        RightShift,RightControl,RightAlt,RightSuper,
        Menu
    } key;

    enum class Action
    {
        Press,
        Release
    } action;

    EventKey(Key key, Action action) : key(key), action(action) {}
};

} // namespace klartraum

#endif // KLARTRAUM_EVENTS_HPP