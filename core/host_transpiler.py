import re


TRIPLE_CHEVRON = re.compile(
    r"\b([A-Za-z_]\w*)\s*<<<\s*(.*?)\s*,\s*(.*?)\s*>>>\s*\((.*?)\)\s*;",
    re.DOTALL,
)
GLOBAL_SIGNATURE = re.compile(r"\b__global__\s+[^;{}]+(?P<end>[;{])", re.MULTILINE)


def transform_host_source(text):
    transformed = _remove_global_code(text)
    transformed = rewrite_triple_chevrons(transformed)
    transformed = re.sub(
        r'^\s*#\s*include\s*[<"](?:cuda_runtime\.h|cuda\.h)[>"]\s*$',
        "",
        transformed,
        flags=re.MULTILINE,
    )
    if "cudaLaunch(" in transformed or re.search(
        r"\bcuda(?:Malloc|Free|Memcpy|DeviceSynchronize|GetLastError)\b", transformed
    ):
        transformed = '#include "cuda_to_metal.h"\n' + transformed.lstrip()
    return transformed


def rewrite_triple_chevrons(text):
    return TRIPLE_CHEVRON.sub(
        lambda match: (
            f"cudaLaunch({match.group(1)}, {match.group(2)}, "
            f"{match.group(3)}{', ' if match.group(4).strip() else ''}{match.group(4)});"
        ),
        text,
    )


def has_host_code(text):
    meaningful = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)
    return bool(meaningful.strip())


def _remove_global_code(text):
    output = []
    cursor = 0
    for match in GLOBAL_SIGNATURE.finditer(text):
        if match.start() < cursor:
            continue
        output.append(text[cursor:match.start()])
        if match.group("end") == ";":
            cursor = match.end()
            continue
        cursor = _matching_brace(text, match.end() - 1) + 1
    output.append(text[cursor:])
    return "".join(output)


def _matching_brace(text, opening):
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return len(text) - 1
