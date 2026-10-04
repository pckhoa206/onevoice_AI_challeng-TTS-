//=============================================================================
//
//  QUALCOMM AI ENGINE DIRECT (QNN) OP PACKAGE SPECIFICATION
//  Custom Operator Package Interface Header for Supertonic Tokenizer
//
//=============================================================================

#ifndef QNN_OP_PACKAGE_H
#define QNN_OP_PACKAGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// QNN Error handle codes
typedef uint32_t Qnn_ErrorHandle_t;
#define QNN_SUCCESS                      0
#define QNN_OP_PACKAGE_ERROR_NONE        0
#define QNN_OP_PACKAGE_ERROR_GENERAL     1
#define QNN_OP_PACKAGE_ERROR_INVALID_ARG 2
#define QNN_OP_PACKAGE_ERROR_MEM_ALLOC   3

// OpPackage Info
typedef struct {
    const char* packageName;
    const char** opNames;
    uint32_t numOps;
    const char* qnnVersion;
    const char* targetBackend; // e.g. "HTP", "CPU"
} QnnOpPackage_Info_t;

// Forward declarations of opaque types
typedef void* QnnOpPackage_GlobalInfrastructure_t;
typedef void* Qnn_OpConfig_t;
typedef void* QnnOpPackage_OpImpl_t;

// QNN Custom Op Function Table interface
typedef struct {
    Qnn_ErrorHandle_t (*init)(QnnOpPackage_GlobalInfrastructure_t infrastructure);
    Qnn_ErrorHandle_t (*terminate)(void);
    Qnn_ErrorHandle_t (*getInfo)(const QnnOpPackage_Info_t** info);
    Qnn_ErrorHandle_t (*validateOpConfig)(Qnn_OpConfig_t opConfig);
    Qnn_ErrorHandle_t (*createOpImpl)(QnnOpPackage_GlobalInfrastructure_t infrastructure,
                                      Qnn_OpConfig_t opConfig,
                                      QnnOpPackage_OpImpl_t* opImpl);
    Qnn_ErrorHandle_t (*freeOpImpl)(QnnOpPackage_OpImpl_t opImpl);
    Qnn_ErrorHandle_t (*executeOpImpl)(QnnOpPackage_OpImpl_t opImpl);
} QnnOpPackage_Interface_t;

// Standard QNN Entry Points
Qnn_ErrorHandle_t QnnOpPackage_initialize(QnnOpPackage_GlobalInfrastructure_t infrastructure);
Qnn_ErrorHandle_t QnnOpPackage_getFunctionTable(const QnnOpPackage_Interface_t** interface);
Qnn_ErrorHandle_t QnnOpPackage_getInfo(const QnnOpPackage_Info_t** info);
Qnn_ErrorHandle_t QnnOpPackage_terminate(void);

// High-speed direct C ABI export for Host / Test Runner execution
int supertonic_qnn_tokenize(const char* utf8_text, const char* lang, int64_t* out_ids, float* out_mask, int max_len);

#ifdef __cplusplus
}
#endif

#endif // QNN_OP_PACKAGE_H
