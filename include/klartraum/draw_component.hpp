#ifndef DRAW_COMPONENT_HPP
#define DRAW_COMPONENT_HPP

namespace klartraum {
class DrawComponent {
public:
    virtual void draw(uint32_t currentFrame) = 0;
};

} // namespace klartraum

#endif // DRAW_COMPONENT_HPP