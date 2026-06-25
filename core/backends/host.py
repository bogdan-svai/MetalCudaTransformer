import re


TRIPLE_CHEVRON = re.compile(
    r"\b([A-Za-z_]\w*)\s*<<<\s*(.*?)\s*>>>\s*\((.*?)\)\s*;",
    re.DOTALL,
)
GLOBAL_SIGNATURE = re.compile(r"\b__global__\s+[^;{}]+(?P<end>[;{])", re.MULTILINE)
DEVICE_SIGNATURE = re.compile(
    r"\b(?:(?:static|inline)\s+)*__device__\s+[^;{}]+(?P<end>[;{])",
    re.MULTILINE,
)
HOST_DEVICE_MARKER = "MCU_HOST_DEVICE_QUALIFIER"


def transform_host_source(text):
    transformed = re.sub(
        r"(?:__host__\s+__device__|__device__\s+__host__)",
        HOST_DEVICE_MARKER,
        text,
    )
    transformed = _remove_qualified_code(transformed, GLOBAL_SIGNATURE)
    transformed = _remove_qualified_code(transformed, DEVICE_SIGNATURE)
    transformed = transformed.replace(HOST_DEVICE_MARKER, "")
    transformed = rewrite_triple_chevrons(transformed)
    transformed = re.sub(
        r'^\s*#\s*include\s*[<"](?:cuda_runtime\.h|cuda\.h)[>"]\s*$',
        "",
        transformed,
        flags=re.MULTILINE,
    )
    if "cudaLaunch(" in transformed or "cudaLaunchShared(" in transformed or re.search(
        r"\bcuda(?:Malloc|Free|Memcpy|MemcpyAsync|Memset|DeviceSynchronize|GetLastError|StreamCreate|StreamDestroy|StreamSynchronize)\b", transformed
    ):
        transformed = '#include "cuda_to_metal.h"\n' + transformed.lstrip()
    return transformed


def rewrite_triple_chevrons(text):
    from core.frontend.regex_parser import split_top_level

    def replace(match):
        configuration = [item.strip() for item in split_top_level(match.group(2))]
        if len(configuration) < 2:
            return match.group(0)
        arguments = match.group(3).strip()
        separator = ", " if arguments else ""
        if len(configuration) >= 4:
            return (
                f"cudaLaunchSharedOnStream({match.group(1)}, {configuration[0]}, {configuration[1]}, "
                f"{configuration[2]}, {configuration[3]}{separator}{arguments});"
            )
        if len(configuration) >= 3:
            return (
                f"cudaLaunchShared({match.group(1)}, {configuration[0]}, {configuration[1]}, "
                f"{configuration[2]}{separator}{arguments});"
            )
        return (
            f"cudaLaunch({match.group(1)}, {configuration[0]}, {configuration[1]}"
            f"{separator}{arguments});"
        )

    return TRIPLE_CHEVRON.sub(replace, text)


def has_host_code(text):
    meaningful = re.sub(r"^\s*#.*$", "", text, flags=re.MULTILINE)
    return bool(meaningful.strip())


def _remove_qualified_code(text, pattern):
    output = []
    cursor = 0
    for match in pattern.finditer(text):
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
