#include <iostream>
#include <google/protobuf/message.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>

int main() {
    std::cout << "ONNX Example - Protobuf Integration Test" << std::endl;
    
    // Verify protobuf library version
    std::cout << "Protobuf version: " << GOOGLE_PROTOBUF_VERSION << std::endl;
    std::cout << "Protobuf version string: " << google::protobuf::internal::VersionString(GOOGLE_PROTOBUF_VERSION) << std::endl;
    
    // Initialize protobuf library
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    
    std::cout << "Protobuf library initialized successfully!" << std::endl;
    std::cout << "Ready for ONNX model loading implementation." << std::endl;
    
    // Clean up protobuf library
    google::protobuf::ShutdownProtobufLibrary();
    
    return 0;
}
