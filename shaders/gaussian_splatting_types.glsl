struct Gaussian {
    vec3 position;
    vec4 rotation;
    vec3 scale;
    vec3 color;
    float alpha;
    float shR[15];
    float shG[15];
    float shB[15];
};

struct Gaussian2D {
    vec2 position; // position in screen space
    float z; // depth value
    uint binMask; // bitmask for which bins this gaussian is in, only one bit should be set
    mat2 covarianceInv; // inverse covariance matrix for the gaussian
    vec3 color; // color of the gaussian
    float alpha;
};

struct StartAndEnd {
    uint start;
    uint end;
};

struct DispatchIndirectCommand {
    uint    x;
    uint    y;
    uint    z;
};