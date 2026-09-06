# CudaMetalTransformer architecture contract

The compiler is an AST/IR compiler. Text processing is not allowed to make
semantic decisions about device code.

## Frontend contract

1. libclang produces the semantic project, including types, calls, effects,
   aliases, address spaces, statements and source locations.
2. A translation unit is accepted only when every device definition has a
   complete semantic record. `MCU_SEMANTIC_STRICT=1` enforces this contract.
3. The legacy parser may provide driver metadata and source compatibility, but
   it must not be used to infer purity, aliasing, memory safety or optimization
   legality.

## IR contract

Passes consume typed IR and return typed IR plus a diagnostic/report. A pass
must be deterministic, idempotent, and conservative when proof is missing.
Every transformed node retains its CUDA source location and an optimization
reason in the report.

## Backend contract

Metal emission is a lowering step, not a pattern-rewrite engine. CUDA concepts
such as pointers, atomics, barriers, address spaces and launch geometry are
lowered from IR metadata. Textual scanning is allowed only for formatting or
for syntax that has already been proven by the frontend; it cannot discover
new symbols or decide whether a transformation is safe.

## Optimization contract

An optimization is enabled only if its preconditions are proven. It must have:

- a typed precondition checker;
- a correctness-preserving fallback;
- a structured report with estimated cost and bytes/registers saved;
- a differential test against the unoptimized result;
- a benchmark showing a real workload benefit.

The migration order is: alias/points-to IR, statement IR, typed loop and array
passes, declarative CUDA-to-Metal lowering, then removal of the compatibility
parser from the device compilation path.

