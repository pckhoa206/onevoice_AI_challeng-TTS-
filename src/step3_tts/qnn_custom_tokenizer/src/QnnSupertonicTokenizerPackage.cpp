//=============================================================================
//
//  QUALCOMM AI ENGINE DIRECT (QNN) OP PACKAGE IMPLEMENTATION
//  Custom Operator Package for Supertonic Fast Tokenizer (English & Korean)
//
//=============================================================================

#include "QnnOpPackage.h"
#include "supertonic_tokenizer_kernel.h"
#include <memory>
#include <cstring>
#include <mutex>

static const char* G_OP_NAMES[] = {
    "SupertonicTokenizerOp"
};

static QnnOpPackage_Info_t G_PACKAGE_INFO = {
    "SupertonicTokenizerOpPackage", // packageName
    G_OP_NAMES,                    // opNames
    1,                             // numOps
    "2.20.0",                      // qnnVersion
    "HTP"                          // targetBackend
};

static std::unique_ptr<supertonic::TokenizerKernel> g_kernel_instance = nullptr;
static std::mutex g_kernel_mutex;

static supertonic::TokenizerKernel* get_kernel() {
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    if (!g_kernel_instance) {
        g_kernel_instance = std::make_unique<supertonic::TokenizerKernel>();
    }
    return g_kernel_instance.get();
}

// QNN OpPackage function table implementations
static Qnn_ErrorHandle_t init_impl(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    (void)infrastructure;
    get_kernel();
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t terminate_impl(void) {
    std::lock_guard<std::mutex> lock(g_kernel_mutex);
    g_kernel_instance.reset();
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t get_info_impl(const QnnOpPackage_Info_t** info) {
    if (!info) return QNN_OP_PACKAGE_ERROR_INVALID_ARG;
    *info = &G_PACKAGE_INFO;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t validate_op_config_impl(Qnn_OpConfig_t opConfig) {
    (void)opConfig;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t create_op_impl(QnnOpPackage_GlobalInfrastructure_t infrastructure,
                                       Qnn_OpConfig_t opConfig,
                                       QnnOpPackage_OpImpl_t* opImpl) {
    (void)infrastructure;
    (void)opConfig;
    if (!opImpl) return QNN_OP_PACKAGE_ERROR_INVALID_ARG;
    *opImpl = reinterpret_cast<void*>(get_kernel());
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t free_op_impl(QnnOpPackage_OpImpl_t opImpl) {
    (void)opImpl;
    return QNN_SUCCESS;
}

static Qnn_ErrorHandle_t execute_op_impl(QnnOpPackage_OpImpl_t opImpl) {
    (void)opImpl;
    // Execution inside QNN Graph Tensor pipeline
    return QNN_SUCCESS;
}

static QnnOpPackage_Interface_t G_QNN_INTERFACE = {
    init_impl,
    terminate_impl,
    get_info_impl,
    validate_op_config_impl,
    create_op_impl,
    free_op_impl,
    execute_op_impl
};

extern "C" {

Qnn_ErrorHandle_t QnnOpPackage_initialize(QnnOpPackage_GlobalInfrastructure_t infrastructure) {
    return init_impl(infrastructure);
}

Qnn_ErrorHandle_t QnnOpPackage_getFunctionTable(const QnnOpPackage_Interface_t** interface) {
    if (!interface) return QNN_OP_PACKAGE_ERROR_INVALID_ARG;
    *interface = &G_QNN_INTERFACE;
    return QNN_SUCCESS;
}

Qnn_ErrorHandle_t QnnOpPackage_getInfo(const QnnOpPackage_Info_t** info) {
    return get_info_impl(info);
}

Qnn_ErrorHandle_t QnnOpPackage_terminate(void) {
    return terminate_impl();
}

/**
 * Direct High-Speed C ABI Export for Host, Android JNI, and Python ctypes.
 */
int supertonic_qnn_tokenize(const char* utf8_text, const char* lang, int64_t* out_ids, float* out_mask, int max_len) {
    auto* kernel = get_kernel();
    if (!kernel) return 0;
    return kernel->tokenize(utf8_text, lang, out_ids, out_mask, max_len);
}

} // extern "C"
