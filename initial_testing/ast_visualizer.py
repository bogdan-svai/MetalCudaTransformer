import os
import re
import sys
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import customtkinter as ctk
from clang.cindex import Index, Config, CursorKind

def configurare_clang():
    paths = [
        "/Library/Developer/CommandLineTools/usr/lib/libclang.dylib",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib"
    ]
    for path in paths:
        if os.path.exists(path):
            Config.set_library_file(path)
            Config.set_compatibility_check(False)
            return True
    return False

class InteractivNodeGraph(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg="#1e1e1e", highlightthickness=0, **kwargs)
        self.nodes = {}  
        self.edges = []  
        self._drag_data = {"x": 0, "y": 0, "item": None}
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.pan_step = 60
        self.configure(takefocus=True)
        
        self.tag_bind("node", "<ButtonPress-1>", self.on_node_press)
        self.tag_bind("node", "<ButtonRelease-1>", self.on_node_release)
        self.tag_bind("node", "<B1-Motion>", self.on_node_motion)
        self.bind("<Button-1>", lambda _event: self.focus_set())
        self.bind("<KeyPress-minus>", lambda _event: self.micsoreaza())
        self.bind("<KeyPress-underscore>", lambda _event: self.micsoreaza())
        self.bind("<KeyPress-equal>", lambda _event: self.mareste())
        self.bind("<KeyPress-plus>", lambda _event: self.mareste())
        self.bind("<KeyPress-0>", lambda _event: self.reseteaza_vizualizarea())
        self.bind("<Left>", lambda _event: self.muta_vizualizarea(self.pan_step, 0))
        self.bind("<Right>", lambda _event: self.muta_vizualizarea(-self.pan_step, 0))
        self.bind("<Up>", lambda _event: self.muta_vizualizarea(0, self.pan_step))
        self.bind("<Down>", lambda _event: self.muta_vizualizarea(0, -self.pan_step))

    def adauga_nod(self, nod_id, label, tip):
        if nod_id not in self.nodes:
            self.nodes[nod_id] = {"x": 0, "y": 0, "label": label, "tip": tip, "tag": f"node_{nod_id}"}

    def adauga_legatura(self, sursa_id, tinta_id, tip_legatura="legatura", eticheta=""):
        legatura = (sursa_id, tinta_id, tip_legatura, eticheta)
        if legatura not in self.edges and sursa_id in self.nodes and tinta_id in self.nodes:
            self.edges.append(legatura)

    def ecran_x(self, x):
        return x * self.zoom + self.offset_x

    def ecran_y(self, y):
        return y * self.zoom + self.offset_y

    def scala_dimensiune(self, valoare, minim=1):
        return max(minim, int(round(valoare * self.zoom)))

    def micsoreaza(self):
        self.zoom = max(0.25, self.zoom * 0.85)
        self.deseneaza_graf()

    def mareste(self):
        self.zoom = min(2.5, self.zoom * 1.15)
        self.deseneaza_graf()

    def reseteaza_vizualizarea(self):
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.deseneaza_graf()

    def muta_vizualizarea(self, dx, dy):
        self.offset_x += dx
        self.offset_y += dy
        self.deseneaza_graf()

    def deseneaza_graf(self):
        self.delete("all")
        for sursa, tinta, tip, eticheta in self.edges:
            if sursa in self.nodes and tinta in self.nodes:
                n1 = self.nodes[sursa]
                n2 = self.nodes[tinta]
                if tip == "cross_file":
                    culoare, grosime = "#ffa500", 3
                elif tip == "apeleaza":
                    culoare, grosime = "#27ae60", 2
                else:
                    culoare, grosime = "#444444", 1.5
                x1, y1 = self.ecran_x(n1["x"]), self.ecran_y(n1["y"])
                x2, y2 = self.ecran_x(n2["x"]), self.ecran_y(n2["y"])
                self.create_line(x1, y1, x2, y2, 
                                 fill=culoare, width=max(1, grosime * self.zoom), arrow=tk.LAST,
                                 arrowshape=(
                                     self.scala_dimensiune(10, 4),
                                     self.scala_dimensiune(12, 5),
                                     self.scala_dimensiune(5, 2)
                                 ))
                if eticheta:
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    text_id = self.create_text(
                        mid_x, mid_y - 10,
                        text=eticheta,
                        fill="#f5f5f5",
                        font=("Helvetica", self.scala_dimensiune(9, 6)),
                        width=self.scala_dimensiune(230, 90)
                    )
                    bbox = self.bbox(text_id)
                    if bbox:
                        pad = self.scala_dimensiune(4, 2)
                        fundal = self.create_rectangle(
                            bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad,
                            fill="#1e1e1e", outline=culoare
                        )
                        self.tag_lower(fundal, text_id)

        for nod_id, date in self.nodes.items():
            x, y = self.ecran_x(date["x"]), self.ecran_y(date["y"])
            r = self.scala_dimensiune(25, 8)  
            if date["tip"] == "CPP_FILE": cul = "#34495e"
            elif date["tip"] == "CUDA_FILE": cul = "#8e44ad"
            elif date["tip"] == "SOURCE_DIR": cul = "#7f8c8d"
            elif date["tip"] == "MAIN_FUNC": cul = "#f1c40f"
            elif date["tip"] == "KERNEL": cul = "#1f85de"    
            elif date["tip"] == "HOST_FUNC": cul = "#2ecc71"  
            else: cul = "#e67e22"                            
            
            self.create_oval(x-r, y-r, x+r, y+r, fill=cul, outline="white",
                             width=self.scala_dimensiune(2, 1), tags=("node", date["tag"]))
            self.create_text(x, y+r+self.scala_dimensiune(12, 5), text=date["label"], fill="white",
                             font=("Helvetica", self.scala_dimensiune(11, 6), "bold"), tags=("node", date["tag"]))
            self.create_text(x, y, text=date["tip"][:4], fill="white",
                             font=("Helvetica", self.scala_dimensiune(9, 6)), tags=("node", date["tag"]))

    def on_node_press(self, event):
        items = self.find_withtag("current")
        if items:
            self._drag_data["item"] = None
            tags = self.get_tags(items[0])
            for nod_id, date in self.nodes.items():
                if date["tag"] in tags:
                    self._drag_data["item"] = nod_id
                    self._drag_data["x"] = event.x
                    self._drag_data["y"] = event.y
                    break

    def on_node_release(self, event):
        self._drag_data["item"] = None

    def on_node_motion(self, event):
        nod_id = self._drag_data["item"]
        if nod_id:
            delta_x = event.x - self._drag_data["x"]
            delta_y = event.y - self._drag_data["y"]
            self.nodes[nod_id]["x"] += delta_x / self.zoom
            self.nodes[nod_id]["y"] += delta_y / self.zoom
            self._drag_data["x"] = event.x
            self._drag_data["y"] = event.y
            self.deseneaza_graf()

class ASTVisualizer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("CUDA AST Multi-Kernel Deep Metadata Inspector")
        self.geometry("1400x850")
        ctk.set_appearance_mode("dark")
        
        configurare_clang()

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=4)
        self.grid_rowconfigure(0, weight=1)
        
        self.frame_stanga = ctk.CTkFrame(self, width=320)
        self.frame_stanga.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        lbl = ctk.CTkLabel(self.frame_stanga, text="AST Metrici GPU", font=("Helvetica", 16, "bold"))
        lbl.pack(pady=10)

        self.toate_fisierele_sursa = self.descopera_toate_fisierele_sursa()
        self.target_curent = self.alege_target_initial()
        self.target_var = ctk.StringVar(value=self.target_curent)
        self.selector_target = ctk.CTkOptionMenu(
            self.frame_stanga,
            values=self.toate_fisierele_sursa or ["."],
            variable=self.target_var,
            command=self.schimba_target
        )
        self.selector_target.pack(pady=(0, 10), padx=10, fill="x")
        
        self.txt_inspect = ctk.CTkTextbox(self.frame_stanga, height=500, width=300, font=("Courier", 12))
        self.txt_inspect.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.graph_canvas = InteractivNodeGraph(self)
        self.graph_canvas.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.definitii_proiect = {} 
        self.function_metadata = {}
        self.kernel_metadata = {}
        self.kernel_analysis = {}
        self.virtual_line_origins = []
        self.mapeaza_proiect()

    def schimba_target(self, target):
        self.target_curent = target
        self.mapeaza_proiect()

    def mapeaza_proiect(self):
        self.graph_canvas.nodes.clear()
        self.graph_canvas.edges.clear()
        self.definitii_proiect = {}
        self.function_metadata = {}
        self.kernel_metadata = {}
        self.kernel_analysis = {}
        self.virtual_line_origins = []

        index = Index.create()
        target_files = self.descopera_fisiere_sursa(self.target_curent)
        
        # Puntea centrală statică hardware
        self.graph_canvas.adauga_nod("func_cudaLaunch", "cudaLaunch()", "BRIDGE")
        self.definitii_proiect["cudaLaunch"] = "func_cudaLaunch"

        self.adauga_structura_proiect_in_graf(target_files)

        # SOLUȚIE UNIFICATĂ: Unim fișierele într-o unitate globală virtuală temporară
        virtual_tu_path = "__transpiler_tu__.cpp"
        try:
            self.scrie_unitate_virtuala(virtual_tu_path, target_files)

            # Mapăm structural ierarhia din Unitatea Virtuală Unificată
            tu = index.parse(virtual_tu_path, args=['-x', 'c++', '-std=c++17'])
            erori = [str(d) for d in tu.diagnostics if d.severity >= d.Error]
            if erori:
                messagebox.showwarning(
                    "Avertisment parsare AST",
                    "Unele fragmente nu au putut fi parsate complet:\n\n" + "\n".join(erori[:6])
                )
            
            self.colecteaza_definitii_globale(tu.cursor)
            self.analizeaza_flux_apeluri(tu.cursor, None, {})
            self.analizeaza_kerneluri_static(target_files)
        finally:
            # Curățăm fișierul temporar după parsare, chiar dacă libclang raportează erori.
            if os.path.exists(virtual_tu_path):
                os.remove(virtual_tu_path)

        self.aranjeaza_straturi_dinamice()
        self.graph_canvas.deseneaza_graf()
        self.afiseaza_raport_ast()

    def descopera_toate_fisierele_sursa(self):
        radacina = Path(".")
        extensii_sursa = {".cpp", ".cu"}
        fisiere = []

        for path in radacina.rglob("*"):
            if not path.is_file() or path.suffix not in extensii_sursa:
                continue

            parti = set(path.parts)
            if "__pycache__" in parti or ".git" in parti:
                continue
            if "test_compiler" in path.name or "__transpiler_tu__" in path.name:
                continue

            fisiere.append(path.as_posix())

        return sorted(fisiere)

    def alege_target_initial(self):
        if len(sys.argv) > 1:
            target_cli = Path(sys.argv[1]).as_posix()
            if target_cli in self.toate_fisierele_sursa:
                return target_cli
            messagebox.showwarning("Target invalid", f"Targetul cerut nu există în proiect:\n{target_cli}")

        if "main.cpp" in self.toate_fisierele_sursa:
            return "main.cpp"
        for candidat in self.toate_fisierele_sursa:
            if Path(candidat).name == "main.cpp":
                return candidat
        return self.toate_fisierele_sursa[0] if self.toate_fisierele_sursa else "."

    def descopera_fisiere_sursa(self, target_initial):
        if target_initial not in self.toate_fisierele_sursa:
            return self.toate_fisierele_sursa

        index_definitii = self.indexeaza_definitii_functii(self.toate_fisierele_sursa)
        fisiere_selectate = set()
        coada = [target_initial]

        while coada:
            fisier = coada.pop(0)
            if fisier in fisiere_selectate:
                continue

            fisiere_selectate.add(fisier)
            continut = self.citeste_text_sursa(fisier)

            for include in self.extrage_includeuri_locale(fisier, continut):
                if include in self.toate_fisierele_sursa and include not in fisiere_selectate:
                    coada.append(include)

            for simbol in self.extrage_simboluri_referite(continut):
                for fisier_definitie in index_definitii.get(simbol, []):
                    if fisier_definitie not in fisiere_selectate:
                        coada.append(fisier_definitie)

        return sorted(fisiere_selectate)

    def citeste_text_sursa(self, fisier):
        try:
            return Path(fisier).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return Path(fisier).read_text(encoding="latin-1")

    def indexeaza_definitii_functii(self, fisiere):
        index_definitii = {}
        pattern_definitie = re.compile(
            r"(?:^|[\n;])\s*(?:__global__\s+|__host__\s+|__device__\s+|static\s+|inline\s+|extern\s+)*"
            r"[\w:<>,~*&\s]+\s+([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
            re.MULTILINE
        )
        cuvinte_rezervate = {"if", "for", "while", "switch", "catch"}

        for fisier in fisiere:
            continut = self.citeste_text_sursa(fisier)
            for match in pattern_definitie.finditer(continut):
                nume = match.group(1)
                if nume not in cuvinte_rezervate:
                    index_definitii.setdefault(nume, set()).add(fisier)

        return {nume: sorted(fisiere_def) for nume, fisiere_def in index_definitii.items()}

    def extrage_includeuri_locale(self, fisier_curent, continut):
        includeuri = []
        director_curent = Path(fisier_curent).parent
        for match in re.finditer(r'^\s*#include\s+"([^"]+)"', continut, re.MULTILINE):
            cale_include_path = director_curent / match.group(1)
            try:
                cale_include = cale_include_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
            except ValueError:
                cale_include = cale_include_path.as_posix()
            includeuri.append(cale_include)
        return includeuri

    def extrage_simboluri_referite(self, continut):
        simboluri = set()
        cuvinte_rezervate = {
            "if", "for", "while", "switch", "return", "sizeof", "catch",
            "dim3", "cudaLaunch"
        }

        for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", continut):
            nume = match.group(1)
            if nume not in cuvinte_rezervate:
                simboluri.add(nume)

        for match in re.finditer(r"\bcudaLaunch\s*\(\s*([A-Za-z_]\w*)", continut):
            simboluri.add(match.group(1))

        return simboluri

    def id_din_cale(self, prefix, cale):
        sigur = cale.replace("\\", "/").replace("/", "__").replace(".", "_").replace(" ", "_")
        return f"{prefix}_{sigur}"

    def id_fisier_sursa(self, fisier):
        return self.id_din_cale("file", fisier)

    def id_director_sursa(self, director):
        if director in ("", "."):
            return "dir_project_root"
        return self.id_din_cale("dir", director)

    def adauga_structura_proiect_in_graf(self, target_files):
        self.graph_canvas.adauga_nod("dir_project_root", ".", "SOURCE_DIR")
        self.graph_canvas.nodes["dir_project_root"]["depth"] = 0

        directoare_adaugate = {"."}
        for fisier in target_files:
            cale = Path(fisier)
            parinte_curent = "dir_project_root"
            parti_director = cale.parts[:-1]

            for index_parti in range(len(parti_director)):
                director = Path(*parti_director[:index_parti + 1]).as_posix()
                id_director = self.id_director_sursa(director)
                if director not in directoare_adaugate:
                    self.graph_canvas.adauga_nod(id_director, parti_director[index_parti], "SOURCE_DIR")
                    self.graph_canvas.nodes[id_director]["depth"] = index_parti + 1
                    directoare_adaugate.add(director)
                self.graph_canvas.adauga_legatura(parinte_curent, id_director, "contine")
                parinte_curent = id_director

            id_fisier = self.id_fisier_sursa(fisier)
            tip_fisier = "CUDA_FILE" if cale.suffix == ".cu" else "CPP_FILE"
            self.graph_canvas.adauga_nod(id_fisier, cale.name, tip_fisier)
            self.graph_canvas.nodes[id_fisier]["depth"] = len(parti_director) + 1
            self.graph_canvas.adauga_legatura(parinte_curent, id_fisier, "contine")

    def scrie_unitate_virtuala(self, virtual_tu_path, target_files):
        self.virtual_line_origins = []
        linie_virtuala = 1
        stuburi = [
            "#define __global__",
            "#define __device__",
            "#define __host__",
            "struct dim3 { unsigned int x; unsigned int y; unsigned int z; dim3(unsigned int vx=1, unsigned int vy=1, unsigned int vz=1): x(vx), y(vy), z(vz) {} };",
            "extern dim3 threadIdx;",
            "extern dim3 blockIdx;",
            "extern dim3 blockDim;",
            "extern dim3 gridDim;",
            "template <typename Kernel, typename... Args> void cudaLaunch(Kernel kernel, dim3 blocks, dim3 threads, Args... args);",
            "template <typename Kernel, typename... Args> void cudaLaunch(Kernel kernel, int blocks, int threads, Args... args);",
        ]

        with open(virtual_tu_path, "w", encoding="utf-8") as f_tu:
            for linie in stuburi:
                f_tu.write(linie + "\n")
                linie_virtuala += 1

            for fisier in target_files:
                f_tu.write(f"\n// === SOURCE_FILE: {fisier} ===\n")
                linie_virtuala += 2
                start = linie_virtuala
                with open(fisier, "r", encoding="utf-8") as f_src:
                    for linie in f_src:
                        if linie.lstrip().startswith("#include"):
                            f_tu.write("\n")
                        else:
                            f_tu.write(linie)
                        linie_virtuala += 1
                self.virtual_line_origins.append((start, linie_virtuala - 1, fisier))

    def fisier_real_cursor(self, cursor):
        if not cursor.location.file:
            return None

        nume_fisier = cursor.location.file.name
        if "__transpiler_tu__" not in os.path.basename(nume_fisier):
            try:
                return Path(nume_fisier).resolve().relative_to(Path.cwd().resolve()).as_posix()
            except ValueError:
                return os.path.basename(nume_fisier)

        linie = cursor.location.line
        for start, end, fisier in self.virtual_line_origins:
            if start <= linie <= end:
                return fisier
        return None

    def colecteaza_definitii_globale(self, cursor):
        if cursor.kind == CursorKind.FUNCTION_DECL and cursor.spelling:
            nume = cursor.spelling
            if cursor.is_definition() and nume != "cudaLaunch":
                orig_file = self.fisier_real_cursor(cursor)
                if not orig_file:
                    return
                
                id_fisier_parinte = self.id_fisier_sursa(orig_file)
                id_nod_func = self.id_din_cale(f"func_{nume}", orig_file)
                tip = self.tip_functie(cursor, nume)
                semnatura_argumente = self.argumente_functie(cursor)
                tip_returnat = cursor.result_type.spelling or "void"
                label = f"{nume}()"
                if tip == "MAIN_FUNC":
                    label = "main()"
                
                self.graph_canvas.adauga_nod(id_nod_func, label, tip)
                adancime_fisier = self.graph_canvas.nodes.get(id_fisier_parinte, {}).get("depth", 1)
                self.graph_canvas.nodes[id_nod_func]["depth"] = adancime_fisier + 1
                self.graph_canvas.adauga_legatura(id_fisier_parinte, id_nod_func, "contine")
                
                self.definitii_proiect[nume] = id_nod_func
                self.definitii_proiect[f"{nume}@{orig_file}"] = id_nod_func
                
                metadata = {
                    "tip_nod": tip,
                    "semnatura_argumente": semnatura_argumente,
                    "return": tip_returnat,
                    "file": orig_file,
                    "gridDim": "Nespecificat",
                    "blockDim": "Nespecificat"
                }
                self.function_metadata[nume] = metadata
                self.function_metadata[f"{nume}@{orig_file}"] = metadata
                if tip == "KERNEL":
                    self.kernel_metadata[nume] = metadata

        for child in cursor.get_children():
            self.colecteaza_definitii_globale(child)

    def argumente_functie(self, cursor):
        return [f"{arg.type.spelling} {arg.spelling}".strip() for arg in cursor.get_arguments()]

    def tip_functie(self, cursor, nume):
        if nume == "main":
            return "MAIN_FUNC"

        tokens = [t.spelling for t in cursor.get_tokens()]
        if "__global__" in tokens:
            return "KERNEL"

        simboluri_cuda = {"threadIdx", "blockIdx", "blockDim", "gridDim"}
        for sub in cursor.walk_preorder():
            if sub.spelling in simboluri_cuda:
                return "KERNEL"

        return "HOST_FUNC"

    def eticheta_semnatura(self, nume_functie):
        metadata = self.function_metadata.get(nume_functie)
        if not metadata:
            return ""

        argumente = ", ".join(metadata["semnatura_argumente"]) or "void"
        return f"{argumente} -> {metadata['return']}"

    def analizeaza_flux_apeluri(self, cursor, id_context_host, variabile_locale):
        id_actual_host = id_context_host
        variabile_context = variabile_locale
        
        if cursor.kind == CursorKind.FUNCTION_DECL and cursor.is_definition() and cursor.spelling != "cudaLaunch":
            orig_file = self.fisier_real_cursor(cursor)
            if orig_file:
                id_actual_host = self.definitii_proiect.get(f"{cursor.spelling}@{orig_file}", self.definitii_proiect.get(cursor.spelling))
                variabile_context = self.colecteaza_variabile_locale(cursor)

        if cursor.kind == CursorKind.CALL_EXPR and id_actual_host:
            if cursor.spelling and cursor.spelling in self.definitii_proiect and cursor.spelling != "cudaLaunch":
                id_tinta = self.definitii_proiect[cursor.spelling]
                self.graph_canvas.adauga_legatura(
                    id_actual_host,
                    id_tinta,
                    "apeleaza",
                    self.eticheta_semnatura(cursor.spelling)
                )

        if cursor.kind == CursorKind.CALL_EXPR and cursor.spelling == "cudaLaunch" and id_actual_host:
            self.graph_canvas.adauga_legatura(id_actual_host, "func_cudaLaunch", "apeleaza", "Kernel, grid, block -> void")
            
            args = list(cursor.get_arguments())
            if len(args) >= 3:
                nume_kernel_detectat = None
                
                # Deep tracking pentru identificarea pointerului către kernel
                for sub_nod in args[0].walk_preorder():
                    if sub_nod.spelling in self.kernel_metadata and self.kernel_metadata[sub_nod.spelling]["tip_nod"] == "KERNEL":
                        nume_kernel_detectat = sub_nod.spelling
                        break
                
                if not nume_kernel_detectat:
                    for t in args[0].get_tokens():
                        if t.spelling in self.kernel_metadata and self.kernel_metadata[t.spelling]["tip_nod"] == "KERNEL":
                            nume_kernel_detectat = t.spelling
                            break

                grid_threads = self.rezolva_expresie_launch(args[1], variabile_context)
                block_threads = self.rezolva_expresie_launch(args[2], variabile_context)

                if nume_kernel_detectat:
                    id_kernel_tinta = self.definitii_proiect[nume_kernel_detectat]
                    self.graph_canvas.adauga_legatura(
                        "func_cudaLaunch",
                        id_kernel_tinta,
                        "cross_file",
                        self.eticheta_semnatura(nume_kernel_detectat)
                    )
                    
                    self.kernel_metadata[nume_kernel_detectat]["gridDim"] = grid_threads
                    self.kernel_metadata[nume_kernel_detectat]["blockDim"] = block_threads

        for child in cursor.get_children():
            self.analizeaza_flux_apeluri(child, id_actual_host, variabile_context)

    def colecteaza_variabile_locale(self, cursor_functie):
        variabile = {}
        for sub in cursor_functie.walk_preorder():
            if sub.kind != CursorKind.VAR_DECL or not sub.spelling:
                continue

            expresie = self.extrage_initializare_variabila(sub)
            if expresie:
                variabile[sub.spelling] = expresie
        return variabile

    def extrage_initializare_variabila(self, cursor_variabila):
        tokens = [t.spelling for t in cursor_variabila.get_tokens()]
        if not tokens or cursor_variabila.spelling not in tokens:
            return ""

        index_nume = tokens.index(cursor_variabila.spelling)
        tip = " ".join(tokens[:index_nume]).strip()
        rest = tokens[index_nume + 1:]
        if not rest:
            return ""

        if rest[0] == "=" and len(rest) > 1:
            return self.formateaza_expresie_tokens(rest[1:])

        if rest[0] == "(" and rest[-1] == ")":
            argumente = self.formateaza_expresie_tokens(rest)
            if tip == "dim3":
                return f"dim3{argumente}"
            return argumente

        return ""

    def rezolva_expresie_launch(self, nod, variabile_locale):
        expresie = self.extrage_text_nod_sau_valoare(nod)
        if expresie in variabile_locale:
            return f"{expresie} = {variabile_locale[expresie]}"
        return expresie

    def analizeaza_kerneluri_static(self, target_files):
        for nume, metadata in self.kernel_metadata.items():
            fisier = metadata.get("file")
            if fisier not in target_files:
                continue

            corp = self.extrage_corp_kernel(fisier, nume)
            analiza = self.analizeaza_corp_kernel(corp, metadata)
            self.kernel_analysis[nume] = analiza
            metadata["analysis"] = analiza

    def extrage_corp_kernel(self, fisier, nume_kernel):
        continut = self.citeste_text_sursa(fisier)
        pattern = re.compile(
            r"__global__\s+[^;{]*\b" + re.escape(nume_kernel) + r"\s*\([^)]*\)\s*\{",
            re.MULTILINE
        )
        match = pattern.search(continut)
        if not match:
            return ""

        start = match.end() - 1
        nivel = 0
        for index in range(start, len(continut)):
            caracter = continut[index]
            if caracter == "{":
                nivel += 1
            elif caracter == "}":
                nivel -= 1
                if nivel == 0:
                    return continut[start + 1:index]
        return continut[start + 1:]

    def elimina_comentarii(self, continut):
        fara_blocuri = re.sub(r"/\*.*?\*/", "", continut, flags=re.DOTALL)
        return re.sub(r"//.*", "", fara_blocuri)

    def analizeaza_corp_kernel(self, corp, metadata):
        corp_curat = self.elimina_comentarii(corp)
        variabile_index = self.extrage_variabile_index(corp_curat)
        acces_memorie = self.extrage_acces_memorie(corp_curat)
        conditii = self.extrage_conditii(corp_curat)
        bucle = self.extrage_bucle(corp_curat)
        simboluri_cuda = self.extrage_simboluri_cuda_kernel(corp_curat)
        axe = self.extrage_axe_cuda(simboluri_cuda)
        launch = self.analizeaza_launch_kernel(metadata, axe)

        nume_variabile_index = set(variabile_index.keys())
        conditii_thread = [
            conditie for conditie in conditii
            if self.expresie_depinde_de_thread(conditie, nume_variabile_index)
        ]

        return {
            "dimensionalitate": self.dimensionalitate_din_axe(axe),
            "axe_cuda": sorted(axe),
            "simboluri_cuda": simboluri_cuda,
            "variabile_index": variabile_index,
            "reads": acces_memorie["reads"],
            "writes": acces_memorie["writes"],
            "conditii": conditii,
            "conditii_thread": conditii_thread,
            "bucle": bucle,
            "features": {
                "uses_shared_memory": "__shared__" in corp_curat,
                "uses_barrier": "__syncthreads" in corp_curat,
                "uses_atomic": bool(re.search(r"\batomic\w+\s*\(", corp_curat)),
                "has_branch": bool(conditii),
                "has_thread_branch": bool(conditii_thread),
                "has_loop": bool(bucle),
            },
            "launch": launch,
            "metal_hints": self.genereaza_metal_hints(axe, launch)
        }

    def extrage_variabile_index(self, corp):
        variabile = {}
        pattern = re.compile(
            r"(?:^|[;\n])\s*(?:const\s+)?(?:int|unsigned|uint|size_t|long|auto)\s+"
            r"([A-Za-z_]\w*)\s*=\s*([^;]*(?:threadIdx|blockIdx|blockDim|gridDim)[^;]*);"
        )
        for match in pattern.finditer(corp):
            variabile[match.group(1)] = self.formateaza_expresie_text(match.group(2).strip())
        return variabile

    def extrage_acces_memorie(self, corp):
        reads = []
        writes = []
        for linie in corp.splitlines():
            linie = linie.strip()
            if not linie:
                continue

            write_match = re.match(r"([A-Za-z_]\w*)\s*\[([^\]]+)\]\s*(?:[+\-*/%&|^]?=)", linie)
            if write_match:
                writes.append(f"{write_match.group(1)}[{self.formateaza_expresie_text(write_match.group(2))}]")

            for match in re.finditer(r"\b([A-Za-z_]\w*)\s*\[([^\]]+)\]", linie):
                acces = f"{match.group(1)}[{self.formateaza_expresie_text(match.group(2))}]"
                if acces not in reads and acces not in writes:
                    reads.append(acces)

        return {"reads": reads, "writes": writes}

    def extrage_conditii(self, corp):
        return [
            self.formateaza_expresie_text(match.group(1))
            for match in re.finditer(r"\bif\s*\(([^)]*)\)", corp)
        ]

    def extrage_bucle(self, corp):
        bucle = []
        for match in re.finditer(r"\b(for|while)\s*\(([^)]*)\)", corp):
            bucle.append(f"{match.group(1)} ({self.formateaza_expresie_text(match.group(2))})")
        return bucle

    def extrage_simboluri_cuda_kernel(self, corp):
        simboluri = []
        for match in re.finditer(r"\b(threadIdx|blockIdx|blockDim|gridDim)(?:\.([xyz]))?", corp):
            simbol = match.group(0)
            if simbol not in simboluri:
                simboluri.append(simbol)
        return simboluri

    def extrage_axe_cuda(self, simboluri):
        axe = set()
        for simbol in simboluri:
            match = re.search(r"\.([xyz])$", simbol)
            if match:
                axe.add(match.group(1))
        return axe

    def dimensionalitate_din_axe(self, axe):
        if "z" in axe:
            return "3D"
        if "y" in axe:
            return "2D"
        if "x" in axe:
            return "1D"
        return "Nedetectata"

    def expresie_depinde_de_thread(self, expresie, variabile_index):
        if re.search(r"\b(threadIdx|blockIdx|blockDim|gridDim)\b", expresie):
            return True
        return any(re.search(r"\b" + re.escape(nume) + r"\b", expresie) for nume in variabile_index)

    def analizeaza_launch_kernel(self, metadata, axe):
        grid_expr = metadata.get("gridDim", "Nespecificat")
        block_expr = metadata.get("blockDim", "Nespecificat")
        block = self.parseaza_dimensiune_launch(block_expr)
        grid = self.parseaza_dimensiune_launch(grid_expr)
        total_threads = block["total"]
        warps = math.ceil(total_threads / 32) if total_threads else None

        avertismente = []
        if any(axa in axe for axa in {"y", "z"}) and block["rank"] == 1:
            avertismente.append("Kernel-ul foloseste axe y/z, dar blockDim detectat este scalar.")
        if any(axa in axe for axa in {"y", "z"}) and grid["rank"] == 1:
            avertismente.append("Kernel-ul foloseste axe y/z, dar gridDim detectat este scalar.")

        return {
            "grid_expr": grid_expr,
            "block_expr": block_expr,
            "grid": grid,
            "block": block,
            "threads_per_block": total_threads,
            "estimated_warps_per_block": warps,
            "warp_size_assumed": 32 if total_threads else None,
            "avertismente": avertismente
        }

    def parseaza_dimensiune_launch(self, expresie):
        text = str(expresie).strip()
        rezultat = {"expr": text, "dims": [], "total": None, "rank": None}
        if not text or text in {"Nespecificat", "Nedetectat"}:
            return rezultat

        text_parse = text.split("=", 1)[1].strip() if "=" in text else text

        dim3_match = re.match(r"dim3\s*\((.*)\)$", text_parse)
        if dim3_match:
            parti = self.imparte_argumente_top_level(dim3_match.group(1))
            dims = []
            for parte in parti:
                if re.fullmatch(r"\d+", parte):
                    dims.append(int(parte))
                else:
                    dims.append(parte)
            while len(dims) < 3:
                dims.append(1)
            rezultat["dims"] = dims[:3]
            rezultat["rank"] = 3 if dims[2] != 1 else 2 if dims[1] != 1 else 1
            if all(isinstance(dim, int) for dim in rezultat["dims"]):
                rezultat["total"] = rezultat["dims"][0] * rezultat["dims"][1] * rezultat["dims"][2]
            return rezultat

        if re.fullmatch(r"\d+", text_parse):
            rezultat["dims"] = [int(text_parse), 1, 1]
            rezultat["total"] = int(text_parse)
            rezultat["rank"] = 1
            return rezultat

        rezultat["dims"] = [text_parse]
        rezultat["rank"] = 1
        return rezultat

    def imparte_argumente_top_level(self, text):
        argumente = []
        curent = []
        nivel = 0
        for caracter in text:
            if caracter == "(":
                nivel += 1
            elif caracter == ")":
                nivel = max(0, nivel - 1)

            if caracter == "," and nivel == 0:
                argumente.append("".join(curent).strip())
                curent = []
            else:
                curent.append(caracter)

        if curent or text.strip():
            argumente.append("".join(curent).strip())
        return argumente

    def genereaza_metal_hints(self, axe, launch):
        hints = [
            "CUDA blockDim -> Metal threads_per_threadgroup",
            "CUDA gridDim -> Metal threadgroups_per_grid"
        ]
        if "x" in axe:
            hints.append("threadIdx.x -> thread_position_in_threadgroup.x")
            hints.append("blockIdx.x -> threadgroup_position_in_grid.x")
        if "y" in axe:
            hints.append("threadIdx.y -> thread_position_in_threadgroup.y")
            hints.append("blockIdx.y -> threadgroup_position_in_grid.y")
        if "z" in axe:
            hints.append("threadIdx.z -> thread_position_in_threadgroup.z")
            hints.append("blockIdx.z -> threadgroup_position_in_grid.z")
        if launch["estimated_warps_per_block"] is not None:
            hints.append("Warp CUDA estimat static; SIMD-group Metal trebuie citit din pipeline la runtime.")
        return hints

    def formateaza_expresie_text(self, expresie):
        return re.sub(r"\s+", " ", expresie).strip()

    def extrage_text_nod_sau_valoare(self, nod):
        tokens = [t.spelling for t in nod.get_tokens()]
        if tokens:
            return self.formateaza_expresie_tokens(tokens)

        if nod.kind == CursorKind.INTEGER_LITERAL:
            tk_list = list(nod.get_tokens())
            if tk_list:
                return tk_list[0].spelling
            
        if nod.kind == CursorKind.DECL_REF_EXPR:
            return nod.spelling

        for sub in nod.walk_preorder():
            if sub.kind == CursorKind.INTEGER_LITERAL:
                tk_list = list(sub.get_tokens())
                if tk_list:
                    return tk_list[0].spelling
            if sub.kind == CursorKind.DECL_REF_EXPR:
                return sub.spelling
        
        return "Nedetectat"

    def formateaza_expresie_tokens(self, tokens):
        rezultat = ""
        fara_spatiu_inainte = {")", "]", "}", ",", ";", ".", "->", "::"}
        fara_spatiu_dupa = {"(", "[", "{", ".", "->", "::"}

        anterior = ""
        for token in tokens:
            if not rezultat:
                rezultat = token
                anterior = token
                continue

            if token in fara_spatiu_inainte or anterior in fara_spatiu_dupa:
                rezultat += token
            else:
                rezultat += " " + token
            anterior = token

        return rezultat

    def afiseaza_raport_ast(self):
        self.txt_inspect.delete("1.0", tk.END)
        self.txt_inspect.insert(tk.END, "=== METADATE EXTRACTATE ===\n\n")
        self.txt_inspect.insert(tk.END, f"Target pornire:\n - {self.target_curent}\n\n")
        self.txt_inspect.insert(tk.END, "Comenzi graf:\n")
        self.txt_inspect.insert(tk.END, " - Click pe graf, apoi +/- pentru zoom\n")
        self.txt_inspect.insert(tk.END, " - Săgeți pentru deplasare, 0 pentru reset\n\n")
        self.txt_inspect.insert(tk.END, "Funcții proiect:\n")
        for nume, date in self.function_metadata.items():
            if "@" in nume:
                continue
            argumente = ", ".join(date["semnatura_argumente"]) or "void"
            self.txt_inspect.insert(tk.END, f" - {nume}({argumente}) -> {date['return']} [{date['tip_nod']}]\n")
            self.txt_inspect.insert(tk.END, f"   {date['file']}\n")

        self.txt_inspect.insert(tk.END, "\nKernel-uri CUDA:\n")
        
        for nume, date in self.kernel_metadata.items():
            if date["tip_nod"] == "KERNEL":
                analiza = date.get("analysis", {})
                launch = analiza.get("launch", {})
                features = analiza.get("features", {})

                self.txt_inspect.insert(tk.END, f"Kernel Detectat: {nume}()\n")
                self.txt_inspect.insert(tk.END, f" Locație: {date['file']}\n")
                self.txt_inspect.insert(tk.END, f" Config Hardware (Lansare):\n")
                self.txt_inspect.insert(tk.END, f"   -> Blocks (Grid): {date['gridDim']}\n")
                self.txt_inspect.insert(tk.END, f"   -> Threads/Block: {date['blockDim']}\n")
                if launch.get("threads_per_block") is not None:
                    self.txt_inspect.insert(tk.END, f"   -> Threads totale/block: {launch['threads_per_block']}\n")
                    self.txt_inspect.insert(tk.END, f"   -> Warp size presupus CUDA: {launch['warp_size_assumed']}\n")
                    self.txt_inspect.insert(tk.END, f"   -> Warps estimate/block: {launch['estimated_warps_per_block']}\n")
                for avertisment in launch.get("avertismente", []):
                    self.txt_inspect.insert(tk.END, f"   [!] {avertisment}\n")
                self.txt_inspect.insert(tk.END, f" Semnătură Parametri AST:\n")
                for arg in date["semnatura_argumente"]:
                    self.txt_inspect.insert(tk.END, f"   [-] {arg}\n")

                if analiza:
                    self.txt_inspect.insert(tk.END, " Analiză execuție statică:\n")
                    self.txt_inspect.insert(tk.END, f"   -> Dimensionalitate: {analiza['dimensionalitate']}\n")
                    self.txt_inspect.insert(tk.END, f"   -> Axe CUDA: {', '.join(analiza['axe_cuda']) or 'Nedetectate'}\n")
                    self.insereaza_lista_raport("   -> Simboluri CUDA", analiza["simboluri_cuda"])
                    self.insereaza_dict_raport("   -> Variabile index", analiza["variabile_index"])
                    self.insereaza_lista_raport("   -> Reads memorie", analiza["reads"])
                    self.insereaza_lista_raport("   -> Writes memorie", analiza["writes"])
                    self.insereaza_lista_raport("   -> Condiții", analiza["conditii"])
                    self.insereaza_lista_raport("   -> Bucle", analiza["bucle"])
                    self.txt_inspect.insert(
                        tk.END,
                        "   -> Features: "
                        f"shared={features.get('uses_shared_memory')}, "
                        f"barrier={features.get('uses_barrier')}, "
                        f"atomic={features.get('uses_atomic')}, "
                        f"branch={features.get('has_branch')}, "
                        f"thread_branch={features.get('has_thread_branch')}, "
                        f"loop={features.get('has_loop')}\n"
                    )
                    self.insereaza_lista_raport("   -> Metal mapping hints", analiza["metal_hints"])
                self.txt_inspect.insert(tk.END, "-"*34 + "\n")

    def insereaza_lista_raport(self, titlu, valori):
        if not valori:
            self.txt_inspect.insert(tk.END, f"{titlu}: -\n")
            return

        self.txt_inspect.insert(tk.END, f"{titlu}:\n")
        for valoare in valori:
            self.txt_inspect.insert(tk.END, f"      - {valoare}\n")

    def insereaza_dict_raport(self, titlu, valori):
        if not valori:
            self.txt_inspect.insert(tk.END, f"{titlu}: -\n")
            return

        self.txt_inspect.insert(tk.END, f"{titlu}:\n")
        for cheie, valoare in valori.items():
            self.txt_inspect.insert(tk.END, f"      - {cheie} = {valoare}\n")

    def aranjeaza_straturi_dinamice(self):
        adancime_maxima_arbore = max(
            (date.get("depth", 0) for date in self.graph_canvas.nodes.values()
             if date["tip"] in {"SOURCE_DIR", "CPP_FILE", "CUDA_FILE"}),
            default=1
        )
        baza_functii = 90 + (adancime_maxima_arbore + 1) * 130
        x_straturi = {
            "MAIN_FUNC": baza_functii,
            "HOST_FUNC": baza_functii + 170,
            "BRIDGE": baza_functii + 390,
            "KERNEL": baza_functii + 610
        }
        y_contoare = {k: 150 for k in x_straturi.keys()}
        y_contoare["BRIDGE"] = 380
        y_pe_adancime = {}

        for nod_id, date in self.graph_canvas.nodes.items():
            tip = date["tip"]
            if tip in {"SOURCE_DIR", "CPP_FILE", "CUDA_FILE"}:
                adancime = date.get("depth", 0)
                cheie_y = f"tree_{adancime}"
                date["x"] = 90 + adancime * 130
                date["y"] = y_pe_adancime.get(cheie_y, 120)
                y_pe_adancime[cheie_y] = date["y"] + 95
                continue

            date["x"] = x_straturi.get(tip, baza_functii + 200)
            date["y"] = y_contoare.get(tip, 200)
            if tip != "BRIDGE":
                y_contoare[tip] += 180

if __name__ == "__main__":
    app = ASTVisualizer()
    app.mainloop()
