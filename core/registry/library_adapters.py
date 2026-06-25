from dataclasses import dataclass


@dataclass(frozen=True)
class LibraryAdapter:
    library: str
    symbol: str
    status: str
    stage: str
    signatures: tuple[str, ...]
    backend: str | None = None
    notes: str = ""

    def to_dict(self):
        return {
            "library": self.library,
            "symbol": self.symbol,
            "status": self.status,
            "stage": self.stage,
            "signatures": list(self.signatures),
            "backend": self.backend,
            "notes": self.notes,
        }


class LibraryAdapterRegistry:
    def __init__(self, adapters):
        self._adapters = {
            (adapter.library, adapter.symbol): adapter
            for adapter in adapters
        }

    def get(self, library, symbol):
        return self._adapters.get((library, symbol))

    def supported_libraries(self):
        return sorted({
            adapter.library
            for adapter in self._adapters.values()
            if adapter.status == "supported"
        })

    def unsupported_libraries(self):
        return sorted({
            adapter.library
            for adapter in self._adapters.values()
            if adapter.status != "supported"
        })

    def to_manifest(self):
        return [
            adapter.to_dict()
            for adapter in sorted(self._adapters.values(), key=lambda item: (item.library, item.symbol))
        ]


def default_library_registry():
    return LibraryAdapterRegistry([
        LibraryAdapter(
            library="cublas_v2.h",
            symbol="cublasCreate",
            status="supported",
            stage="11B",
            signatures=("cublasStatus_t(cublasHandle_t*)",),
            backend="mcu_runtime",
        ),
        LibraryAdapter(
            library="cublas_v2.h",
            symbol="cublasDestroy",
            status="supported",
            stage="11B",
            signatures=("cublasStatus_t(cublasHandle_t)",),
            backend="mcu_runtime",
        ),
        LibraryAdapter(
            library="cublas_v2.h",
            symbol="cublasSetStream",
            status="supported",
            stage="11B",
            signatures=("cublasStatus_t(cublasHandle_t,cudaStream_t)",),
            backend="mcu_runtime",
        ),
        LibraryAdapter(
            library="cublas_v2.h",
            symbol="cublasGetStream",
            status="supported",
            stage="11B",
            signatures=("cublasStatus_t(cublasHandle_t,cudaStream_t*)",),
            backend="mcu_runtime",
        ),
        LibraryAdapter(
            library="cublas_v2.h",
            symbol="cublasSgemm",
            status="supported",
            stage="11C",
            signatures=("cublasStatus_t(cublasHandle_t,cublasOperation_t,cublasOperation_t,int,int,int,const float*,const float*,int,const float*,int,const float*,float*,int)",),
            backend="mps_matrix_multiplication",
            notes="CUBLAS_OP_N/T, host pointer mode, float32 column-major matrices.",
        ),
        LibraryAdapter(
            library="curand.h",
            symbol="*",
            status="planned",
            stage="future",
            signatures=(),
            notes="Generator semantics and reproducibility require a separate stage.",
        ),
        LibraryAdapter(
            library="cufft.h",
            symbol="*",
            status="planned",
            stage="future",
            signatures=(),
            notes="Plan lifecycle and stride semantics are not implemented in Stage 11.",
        ),
        LibraryAdapter(
            library="cusolverDn.h",
            symbol="*",
            status="unsupported",
            stage="future",
            signatures=(),
        ),
        LibraryAdapter(
            library="cusparse.h",
            symbol="*",
            status="unsupported",
            stage="future",
            signatures=(),
        ),
    ])
