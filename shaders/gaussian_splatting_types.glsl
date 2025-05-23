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
    vec2 position;
    float z;
    uint binMask;
    mat2 covariance;
};

struct StartAndEnd {
    uint start;
    uint end;
};