#include <iostream>
#include <fstream>
#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

// Include generated ONNX headers (will be available after copying proto files)
#include "onnx.pb.h"

// WARNING: AI CODE
bool loadOnnxModel(const std::string& filename, onnx::ModelProto& model) {
    std::ifstream input(filename, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }
    
    // Parse the protobuf from the file
    if (!model.ParseFromIstream(&input)) {
        std::cerr << "Error: Failed to parse ONNX model from " << filename << std::endl;
        return false;
    }
    
    input.close();
    return true;
}

// WARNING: AI CODE
void printModelInfo(const onnx::ModelProto& model) {
    std::cout << "\n=== ONNX Model Information ===" << std::endl;
    
    // Model metadata
    // if (model.has_metadata_props()) {
    //     std::cout << "Model metadata properties count: " << model.metadata_props_size() << std::endl;
    // }
    
    // IR version
    if (model.has_ir_version()) {
        std::cout << "IR Version: " << model.ir_version() << std::endl;
    }
    
    // Producer name and version
    if (model.has_producer_name()) {
        std::cout << "Producer: " << model.producer_name();
        if (model.has_producer_version()) {
            std::cout << " v" << model.producer_version();
        }
        std::cout << std::endl;
    }
    
    // Model version
    if (model.has_model_version()) {
        std::cout << "Model Version: " << model.model_version() << std::endl;
    }
    
    // Domain
    if (model.has_domain()) {
        std::cout << "Domain: " << model.domain() << std::endl;
    }
    
    // Graph information
    if (model.has_graph()) {
        const onnx::GraphProto& graph = model.graph();
        std::cout << "Graph name: " << graph.name() << std::endl;
        std::cout << "Inputs: " << graph.input_size() << std::endl;
        std::cout << "Outputs: " << graph.output_size() << std::endl;
        std::cout << "Nodes: " << graph.node_size() << std::endl;
        std::cout << "Initializers: " << graph.initializer_size() << std::endl;
        
        // Print input information
        std::cout << "\nInput tensors:" << std::endl;
        for (int i = 0; i < graph.input_size(); ++i) {
            const onnx::ValueInfoProto& input = graph.input(i);
            std::cout << "  " << i << ": " << input.name();
            if (input.has_type() && input.type().has_tensor_type()) {
                const onnx::TypeProto::Tensor& tensor_type = input.type().tensor_type();
                if (tensor_type.has_elem_type()) {
                    std::cout << " (type: " << tensor_type.elem_type() << ")";
                }
                if (tensor_type.has_shape()) {
                    std::cout << " shape: [";
                    for (int j = 0; j < tensor_type.shape().dim_size(); ++j) {
                        if (j > 0) std::cout << ", ";
                        const onnx::TensorShapeProto::Dimension& dim = tensor_type.shape().dim(j);
                        if (dim.has_dim_value()) {
                            std::cout << dim.dim_value();
                        } else if (dim.has_dim_param()) {
                            std::cout << dim.dim_param();
                        } else {
                            std::cout << "?";
                        }
                    }
                    std::cout << "]";
                }
            }
            std::cout << std::endl;
        }
        
        // Print output information
        std::cout << "\nOutput tensors:" << std::endl;
        for (int i = 0; i < graph.output_size(); ++i) {
            const onnx::ValueInfoProto& output = graph.output(i);
            std::cout << "  " << i << ": " << output.name();
            if (output.has_type() && output.type().has_tensor_type()) {
                const onnx::TypeProto::Tensor& tensor_type = output.type().tensor_type();
                if (tensor_type.has_elem_type()) {
                    std::cout << " (type: " << tensor_type.elem_type() << ")";
                }
                if (tensor_type.has_shape()) {
                    std::cout << " shape: [";
                    for (int j = 0; j < tensor_type.shape().dim_size(); ++j) {
                        if (j > 0) std::cout << ", ";
                        const onnx::TensorShapeProto::Dimension& dim = tensor_type.shape().dim(j);
                        if (dim.has_dim_value()) {
                            std::cout << dim.dim_value();
                        } else if (dim.has_dim_param()) {
                            std::cout << dim.dim_param();
                        } else {
                            std::cout << "?";
                        }
                    }
                    std::cout << "]";
                }
            }
            std::cout << std::endl;
        }
        
        // Print first few node operations
        std::cout << "\nFirst few operations:" << std::endl;
        int nodes_to_show = std::min(15, graph.node_size());
        for (int i = 0; i < nodes_to_show; ++i) {
            const onnx::NodeProto& node = graph.node(i);
            std::cout << "  " << i << ": " << node.op_type();
            if (node.has_name() && !node.name().empty()) {
                std::cout << " (" << node.name() << ")";
            }
            std::cout << " - inputs: " << node.input_size() << ", outputs: " << node.output_size() << std::endl;
        }
        if (graph.node_size() > nodes_to_show) {
            std::cout << "  ... and " << (graph.node_size() - nodes_to_show) << " more nodes" << std::endl;
        }
    }
}

int main() {
    std::cout << "ONNX Example - Loading Super Resolution Model" << std::endl;
    
    // Verify protobuf library version
    std::cout << "Protobuf version: " << GOOGLE_PROTOBUF_VERSION << std::endl;
    std::cout << "Protobuf version string: " << google::protobuf::internal::VersionString(GOOGLE_PROTOBUF_VERSION) << std::endl;
    
    // Initialize protobuf library
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    std::cout << "Protobuf library initialized successfully!" << std::endl;
    
    // Load the ONNX model
    onnx::ModelProto model;
    const std::string model_path = "C:\\Users\\dfort\\Desktop\\workspace\\super_resolution\\super-resolution-10.onnx";
    
    std::cout << "\nLoading ONNX model from: " << model_path << std::endl;
    
    if (loadOnnxModel(model_path, model)) {
        std::cout << "ONNX model loaded successfully!" << std::endl;
        printModelInfo(model);
    } else {
        std::cerr << "Failed to load ONNX model!" << std::endl;
        return 1;
    }
    
    // Clean up protobuf library
    google::protobuf::ShutdownProtobufLibrary();
    
    return 0;
}
