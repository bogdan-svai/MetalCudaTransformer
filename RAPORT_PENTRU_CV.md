# Raport tehnic pentru folosirea proiectului în CV

## Fișa proiectului

- **Nume:** CudaMetalTransformer / MCU
- **Tip:** prototip de compilator și strat de compatibilitate CUDA-to-Metal
- **Scop:** rularea pe Apple Silicon a unor workload-uri GPU educaționale scrise pentru CUDA, fără un mediu NVIDIA
- **Versiune raportată de CLI:** 0.14.0; ABI runtime 6
- **Stadiu:** prototip avansat, nu înlocuitor complet pentru CUDA
- **Platformă țintă:** macOS / Apple Silicon / Metal

## Rezumat executiv

Proiectul implementează un pipeline care analizează cod CUDA/C++, construiește o reprezentare intermediară tipată și generează Metal Shading Language, apoi compilează shader-ele și aplicația host într-un executabil macOS. Arhitectura declarată este:

```text
CUDA -> libclang AST -> typed semantic IR -> validation/optimization passes
     -> Metal lowering -> Metal runtime -> native macOS executable
```

Nu este doar un set de înlocuiri textuale. IR-ul modelează funcții device, graf de apeluri, tipuri, alias-uri de pointeri, efecte asupra memoriei, spații de adresă, instrucțiuni și locații în sursa CUDA. Totuși, proiectul păstrează un parser regex de compatibilitate pentru anumite cazuri, iar documentația recunoaște că migrarea completă către Statement IR este încă în lucru.

## Ce a fost implementat

### Frontend și compilator

- analiză CUDA/C++ cu libclang și fallback controlat pentru compatibilitate;
- IR semantic tipat pentru simboluri, parametri, apeluri, tipuri, efecte de memorie, alias-uri și statement-uri;
- device linker pentru proiecte cu mai multe fișiere și funcții `__device__`;
- validări conservative pentru memory model, pointeri, atomics, bariere divergente și warp intrinsics;
- păstrarea locațiilor din sursa CUDA în diagnosticele generate;
- lowering pentru kernel-uri, geometrie 1D/2D/3D și spațiile Metal `device`, `threadgroup`, `thread`, `constant`;
- thread coarsening configurabil (factori 1, 2, 4 și 8);
- analiză/optimizare pentru bucle statice, date invariabile, array scalarization, recurențe și acces global la memorie;
- rapoarte structurate ale optimizărilor și mod strict prin `MCU_SEMANTIC_STRICT=1`.

### Toolchain și developer experience

- CLI cu comenzile `check`, `build`, `analyze`, `capabilities`, `compile`, `run`, `make` și un driver compatibil parțial cu `nvcc`;
- compilare separată și link, obiecte MCU, manifest de simboluri și routing table;
- interceptarea proiectelor Make existente printr-un shim `nvcc`;
- suport pentru flag-uri uzuale: `-c`, `-dc`, `-o`, `-I`, `-D`, `-U`, niveluri `-O`, dependențe `-M*`, precum și `-L/-l` la link;
- cache pentru pipeline-uri Metal și build-uri native, plus metrici pe faze;
- contract JSON stabil pentru tooling, diagnostice și coduri de ieșire;
- hook pre-push și GitHub Actions pe runner macOS.

### Runtime CUDA compatibil pe Metal

- management de device și memorie: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaMemset`;
- streams și operații asincrone: create/destroy/synchronize, memcpy/memset async;
- events, elapsed time, device properties, occupancy query și error-state APIs;
- shared memory statică și dinamică, sincronizare de threadgroup și atomics întregi tipate (`Add`, `Min`, `Max`, `Exch`, `CAS`);
- warp subset: shuffle, shuffle-down, any/all, active mask și syncwarp pentru masca completă / width 32;
- subset CUDA Graphs: noduri kernel, memcpy 1D, memset pe bytes, replay și captură pe un singur stream;
- adaptor cu subset cuBLAS, inclusiv SGEMM float32 column-major realizat prin Metal Performance Shaders.

### Testare și performanță

- 131 funcții de test declarate și 132 cazuri colectate în rularea curentă;
- teste unitare, integrare, compilare nativă și comparații diferențiale cu rezultate CPU/oracle;
- workload-uri STREAM, HotSpot, reduction, histogram și SHA-256;
- benchmark matrix pentru baseline, constant promotion, loop unrolling și configurație combinată;
- quality gate configurabil pentru corectitudine și regresii de performanță;
- rapoarte benchmark JSON și Markdown publicate ca artefacte CI.
- suita completă validată local pe macOS cu Metal Toolchain 17A324: **134 passed** după adăugarea testelor de regresie pentru cache-ul de build și invalidarea cache-ului de surse;
- cache no-op stabilizat prin excluderea telemetriei volatile din cheia executabilului și compilare paralelă a unităților host;
- cache comun pentru citirea surselor, invalidat sigur prin metadata fișierului, și index de funcții pentru lookup rapid în IR;
- într-un build reprezentativ al exemplului multi-file, rularea identică la cald a scăzut de la aproximativ **3,23 s la 0,062 s**, cu hit complet pe executabil (măsurare locală, dependentă de sistem);
- quality gate HotSpot validat cu trei probe și două warmup-uri, fără erori de corectitudine sau regresii peste pragul de 20%.

## Dimensiune și tehnologii

Analiza locală a directoarelor `core`, `runtime`, `cli`, `tests`, `scripts`, `benchmarks` și `examples` indică aproximativ **18.700 linii** în fișiere de cod relevante:

- Python: aproximativ 13.562 linii;
- Objective-C++ (`.mm`): aproximativ 2.692 linii;
- CUDA C/C++ (`.cu`): aproximativ 1.491 linii;
- C/C++ headers și surse: aproximativ 935 linii;
- shell tooling: aproximativ 197 linii.

Tehnologii: Python 3, libclang/Clang AST, CUDA C++, Metal Shading Language, Objective-C++, Apple Metal, Metal Performance Shaders, pytest, Make, GitHub Actions, JSON și shell scripting.

## Elemente tehnice valoroase pentru interviu

1. **Compiler design:** separarea frontend/IR/passes/backend și decizii semantice conservative.
2. **Cross-platform GPU systems:** maparea modelului CUDA pe Metal, inclusiv geometrie, memory spaces, atomics și sincronizare.
3. **Runtime engineering:** registru de buffere, scheduler de streams, cache de pipeline-uri, dispatcher de kernel-uri și metrici.
4. **Toolchain compatibility:** emularea unui subset `nvcc`, compilare separată, link și integrarea cu Make.
5. **Correctness engineering:** diagnostice cu source mapping, oracles CPU, teste diferențiale și quality gates.
6. **Performance engineering:** coarsening, loop transformations, memory/register-pressure analysis și benchmarkuri reproductibile.

## Formulări recomandate pentru CV

### Variantă scurtă, în română

**CudaMetalTransformer — compilator CUDA-to-Metal pentru Apple Silicon**  
Am proiectat și implementat un prototip de compilator care transformă aplicații CUDA/C++ în Metal, folosind libclang, IR semantic tipat, validări conservative și un runtime nativ Objective-C++/Metal. Am realizat suport pentru proiecte multi-file, un driver compatibil cu un subset `nvcc`, memory/stream/event APIs, atomics, shared memory, warp intrinsics, un subset CUDA Graphs și integrare cu MPS pentru SGEMM. Am construit o suită pytest și benchmarkuri diferențiale cu quality gates și rapoarte CI.

### Variantă pe bullets, în engleză

- Designed and implemented a CUDA-to-Metal compiler prototype for Apple Silicon using Python, libclang, typed semantic IR, Metal Shading Language, and an Objective-C++ runtime.
- Built multi-file device linking, source-mapped diagnostics, conservative memory/alias validation, configurable thread coarsening, and IR-guided optimization passes.
- Implemented a partial CUDA runtime/toolchain compatibility layer covering memory management, streams, events, typed atomics, shared memory, selected warp intrinsics, CUDA Graphs, and cuBLAS SGEMM via MPS.
- Developed an `nvcc`-compatible driver subset, Make integration, deterministic artifact manifests, build caching, structured metrics, pytest coverage, and macOS CI performance/correctness gates.

### Descriere de o propoziție

Built an end-to-end CUDA-to-Metal compiler and runtime prototype that ports educational GPU workloads to Apple Silicon through semantic analysis, typed IR, native Metal code generation, and CUDA API compatibility.

## Afirmații care trebuie evitate sau validate înainte de CV

- Nu afirma „full CUDA compatibility”; suprafața este explicit parțială.
- Nu afirma paritate de performanță cu NVIDIA/CUDA; proiectul însuși exclude această concluzie.
- Nu publica un procent de speedup fără raportul benchmark, configurația hardware și workload-ul asociat.
- Suita poate fi prezentată ca validată integral în mediul local curent: la 6 septembrie 2026, rularea cu Metal Toolchain 17A324 a produs **136 passed în 56,66 secunde**. Păstrează comanda și mediul atunci când citezi rezultatul.
- Stage 16 adaugă batching pentru stream-uri explicite, pool bounded pentru staging buffers și cache persistent pentru semantic IR. Nu prezenta aceste rezultate ca benchmark universal; menționează Apple M4 Max și workload-ul măsurat.

## Evaluarea stării repository-ului înainte de publicare

Repository-ul are istoric coerent pentru Stage 1-14, în 16 commit-uri principale vizibile, însă working tree-ul este foarte murdar: există modificări tracked, fișiere sursă/test noi și artefacte generate neversionate. Înainte de publicare sunt recomandate:

1. păstrarea în documentație a comenzii validate pentru suita nativă: `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023 python -m pytest -q`;
2. separarea surselor Stage 15 de binare, cache-uri și artefacte generate;
3. eliminarea din Git a `.DS_Store`, `__pycache__`, `.pytest_cache`, executabilelor, `*.mcu/`, `*.air`, `*.metallib`, module-cache și arhivelor generate;
4. extinderea `.gitignore` și verificarea fișierelor deja urmărite;
5. actualizarea README/CI astfel încât etapele și comenzile descrise să corespundă suitei reale;
6. commit-uri tematice mici: semantic IR, optimizer, benchmark suite, CI/docs;
7. adăugarea unei licențe și, dacă repository-ul devine public, verificarea provenienței benchmarkurilor marcate cu NOTICE.

## Prompt gata de trimis unui alt LLM pentru generarea CV-ului

```text
Folosește raportul tehnic de mai jos pentru a redacta secțiunea Projects din CV.
Nu inventa metrici, speedup-uri, utilizatori sau compatibilitate completă. Evidențiază
compiler engineering, GPU systems, typed IR, CUDA-to-Metal lowering, runtime/toolchain
compatibility, testing și CI. Generează:
1) un titlu și o descriere de o linie;
2) 3-4 bullets ATS-friendly, fiecare de maximum 28 de cuvinte;
3) lista tehnologiilor;
4) o variantă în română și una în engleză.
Nivel candidat: [student/junior/mid]. Rol țintă: [Compiler/GPU/Systems/Software Engineer].
Păstrează nuanța că este un prototip cu suport CUDA parțial.

[LIPEȘTE AICI RAPORTUL]
```

## Metodologia raportului

Raportul a fost obținut prin inspecția codului, README-ului, documentelor de arhitectură și Stage 1-16, istoricului și stării Git, contractului `mcu capabilities`, configurației CI și suitei pytest. Componenta oficială Metal Toolchain 17A324 a fost instalată prin Xcode, iar suita finală a fost validată cu `TOOLCHAINS=com.apple.dt.toolchain.Metal.32023 .venv/bin/python -m pytest -q`, rezultând 136 de teste trecute. Au fost implementate și validate optimizări pentru cheia cache, lookup-ul simbolurilor, citirea surselor, compilarea paralelă a unităților host, batching de command buffers, reutilizarea staging buffers și cache semantic persistent.
