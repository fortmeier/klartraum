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

} // namespace klartraum

#endif // KLARTRAUM_EVENTS_HPP