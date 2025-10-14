layout(push_constant) uniform PushConstants {
    uint dims;
    uint dimInput[8];
    uint dimOutput[8];
    uint dimPerm[8];
};
