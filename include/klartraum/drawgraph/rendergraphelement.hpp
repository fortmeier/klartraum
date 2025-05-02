#ifndef KLARTRAUM_RENDERGRAPHELEMENT_HPP
#define KLARTRAUM_RENDERGRAPHELEMENT_HPP

#include "klartraum/drawgraph/drawgraphelement.hpp"
#include "klartraum/drawgraph/uniformbufferobject.hpp"

namespace klartraum {

typedef UniformBufferObjectNew<CameraMVP> CameraUboType;

class RenderGraphElement : public virtual DrawGraphElement {
public:
    std::shared_ptr<CameraUboType> getCameraUBO() {
        for (auto& input : inputs) {
            auto cameraUbo = std::dynamic_pointer_cast<CameraUboType>(input.second);
            if (cameraUbo) {
                return cameraUbo;
            }
        }
        throw std::runtime_error("No CameraUboType found in inputs!");
    }

protected:

};

} // namespace klartraum

#endif // KLARTRAUM_RENDERGRAPHELEMENT_HPP