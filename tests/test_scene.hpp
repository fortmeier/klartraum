#ifndef KLARTRAUM_TESTS_TEST_SCENE_HPP
#define KLARTRAUM_TESTS_TEST_SCENE_HPP

// The Gaussian splatting scene shared by the tests: a stone lantern on moss,
// captured for Klartraum (data/lantern.spz, CC BY 4.0), and the camera that
// frames it.

#include <algorithm>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"

namespace klartraum::test_scene {

// Relative to the repository root, where the tests are run from.
inline const std::string kLanternPath = "data/lantern.spz";

struct Scene {
    std::vector<Gaussian3D> gaussians;
    // Per-axis median of the Gaussian positions, so a few stray splats do not
    // pull the camera pivot off the lantern.
    glm::vec3 centre{0.0f};
};

// The capture is stored with Y pointing down, so it is loaded with flipY.
inline Scene loadLantern() {
    Scene scene;
    scene.gaussians = loadGaussiansSpz(kLanternPath, true);
    std::vector<float> values(scene.gaussians.size());
    for (int axis = 0; axis < 3 && !values.empty(); ++axis) {
        for (size_t i = 0; i < scene.gaussians.size(); ++i) {
            values[i] = scene.gaussians[i].position[axis];
        }
        auto middle = values.begin() + values.size() / 2;
        std::nth_element(values.begin(), middle, values.end());
        scene.centre[axis] = *middle;
    }
    return scene;
}

// Looks at the lantern from the side and slightly above, close enough that the
// lantern and the ground around it span the whole image width; only the upper
// corners show the black background.
inline void frameLantern(InterfaceCameraOrbit& orbit, const Scene& scene) {
    // The orbit camera translates the world by its position before rotating it
    // about the origin, so -centre puts the scene centre at the pivot.
    orbit.setPosition(-scene.centre);
    orbit.setAzimuth(0.9f);
    orbit.setElevation(-0.5f);
    orbit.setDistance(0.35f);
}

} // namespace klartraum::test_scene

#endif // KLARTRAUM_TESTS_TEST_SCENE_HPP
