from dataclasses import dataclass, field
from pathlib import Path
import re
import shlex


SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu"}
OBJECT_SUFFIXES = {".o", ".obj", ".mcuobj", ".a"}
DEPENDENCY_FLAGS = {"-M", "-MM", "-MD", "-MMD"}
VALUE_FLAGS = {"-MF": "dependency_file", "-MT": "dependency_target"}


class NvccInvocationError(ValueError):
    def __init__(self, code, message, argument=None):
        super().__init__(message)
        self.code = code
        self.argument = argument

    def diagnostic(self):
        return {
            "code": self.code,
            "severity": "error",
            "message": str(self),
            "file": None,
            "line": None,
            "column": None,
            "argument": self.argument,
            "phase": "driver",
        }


@dataclass
class NvccInvocation:
    arguments: list[str]
    working_directory: str
    mode: str = "link"
    inputs: list[str] = field(default_factory=list)
    output: str | None = None
    include_paths: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    undefinitions: list[str] = field(default_factory=list)
    optimization: str = "-O2"
    optimization_explicit: bool = False
    host_flags: list[str] = field(default_factory=list)
    library_paths: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    dependency_mode: str | None = None
    dependency_file: str | None = None
    dependency_target: str | None = None

    @property
    def sources(self):
        return [item for item in self.inputs if Path(item).suffix.lower() in SOURCE_SUFFIXES]

    @property
    def objects(self):
        return [item for item in self.inputs if Path(item).suffix.lower() in OBJECT_SUFFIXES]

    def to_dict(self):
        return {
            "arguments": self.arguments,
            "working_directory": self.working_directory,
            "mode": self.mode,
            "inputs": self.inputs,
            "output": self.output,
            "include_paths": self.include_paths,
            "definitions": self.definitions,
            "undefinitions": self.undefinitions,
            "optimization": self.optimization,
            "optimization_explicit": self.optimization_explicit,
            "host_flags": self.host_flags,
            "library_paths": self.library_paths,
            "libraries": self.libraries,
            "dependency_mode": self.dependency_mode,
            "dependency_file": self.dependency_file,
            "dependency_target": self.dependency_target,
        }


def parse_nvcc_arguments(arguments, working_directory=None):
    cwd = Path(working_directory or Path.cwd()).resolve()
    invocation = NvccInvocation(list(arguments), str(cwd))
    compile_only = False
    device_compile = False
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--":
            index += 1
            continue
        if argument in {"-c", "--compile"}:
            compile_only = True
        elif argument in {"-dc", "--device-c"}:
            compile_only = True
            device_compile = True
        elif argument == "-o":
            value, index = _next_value(arguments, index, argument)
            invocation.output = _resolve_output(value, cwd)
        elif argument.startswith("-o") and len(argument) > 2:
            invocation.output = _resolve_output(argument[2:], cwd)
        elif argument in {"-I", "-isystem", "-D", "-U"}:
            value, index = _next_value(arguments, index, argument)
            _append_option(invocation, argument, value, cwd)
        elif argument.startswith("-isystem") and len(argument) > len("-isystem"):
            _append_option(invocation, "-isystem", argument[len("-isystem"):], cwd)
        elif argument[:2] in {"-I", "-D", "-U"} and len(argument) > 2:
            _append_option(invocation, argument[:2], argument[2:], cwd)
        elif argument in {"-O0", "-O1", "-O2", "-O3"}:
            invocation.optimization = argument
            invocation.optimization_explicit = True
        elif argument == "-Xcompiler":
            value, index = _next_value(arguments, index, argument)
            invocation.host_flags.extend(_validated_host_flags(value))
        elif argument.startswith("-Xcompiler="):
            invocation.host_flags.extend(_validated_host_flags(argument.split("=", 1)[1]))
        elif argument in {"-L", "-l"}:
            value, index = _next_value(arguments, index, argument)
            _append_link_option(invocation, argument, value, cwd)
        elif argument.startswith("-L") and len(argument) > 2:
            _append_link_option(invocation, "-L", argument[2:], cwd)
        elif argument.startswith("-l") and len(argument) > 2:
            _append_link_option(invocation, "-l", argument[2:], cwd)
        elif argument in DEPENDENCY_FLAGS:
            invocation.dependency_mode = argument
        elif argument in VALUE_FLAGS:
            value, index = _next_value(arguments, index, argument)
            setattr(invocation, VALUE_FLAGS[argument], _resolve_output(value, cwd))
        elif argument in {"--version", "-V"}:
            raise NvccInvocationError("NvccInformationRequest", "NVCC version output is not an input compilation.", argument)
        elif argument.startswith("-"):
            raise NvccInvocationError(
                "UnsupportedNvccFlag",
                f"Unsupported NVCC flag: {argument}",
                argument,
            )
        else:
            suffix = Path(argument).suffix.lower()
            if suffix not in SOURCE_SUFFIXES | OBJECT_SUFFIXES:
                raise NvccInvocationError(
                    "InvalidNvccInvocation",
                    f"Unsupported NVCC input: {argument}",
                    argument,
                )
            invocation.inputs.append(str((cwd / argument).resolve()) if not Path(argument).is_absolute() else str(Path(argument).resolve()))
        index += 1

    if not invocation.inputs:
        raise NvccInvocationError("InvalidNvccInvocation", "NVCC invocation has no input files.")
    invocation.mode = "device_compile" if device_compile else "compile" if compile_only else "link"
    if invocation.mode != "link" and (invocation.library_paths or invocation.libraries):
        raise NvccInvocationError(
            "InvalidNvccInvocation",
            "Library search and link flags are valid only during an NVCC link invocation.",
        )
    if invocation.mode != "link" and len(invocation.sources) != 1:
        raise NvccInvocationError(
            "InvalidNvccInvocation",
            "Compile-only NVCC invocations require exactly one source input.",
        )
    if not invocation.output:
        source = Path(invocation.sources[0] if invocation.sources else invocation.inputs[0])
        default = source.with_suffix(".o" if invocation.mode != "link" else "")
        invocation.output = str(default)
    return invocation


def _next_value(arguments, index, flag):
    if index + 1 >= len(arguments):
        raise NvccInvocationError("InvalidNvccInvocation", f"Missing value for {flag}.", flag)
    return arguments[index + 1], index + 1


def _append_option(invocation, flag, value, cwd):
    if flag in {"-I", "-isystem"}:
        path = Path(value)
        invocation.include_paths.append(str((cwd / path).resolve() if not path.is_absolute() else path.resolve()))
    elif flag == "-D":
        invocation.definitions.append(value)
    else:
        invocation.undefinitions.append(value)


def _resolve_output(value, cwd):
    path = Path(value)
    return str((cwd / path).resolve() if not path.is_absolute() else path.resolve())


def _validated_host_flags(value):
    flags = shlex.split(value.replace(",", " "))
    allowed = {"-Wall", "-Wextra", "-Werror", "-g", "-pthread", "-fPIC"}
    for flag in flags:
        if flag not in allowed and not flag.startswith("-Wno-"):
            raise NvccInvocationError(
                "UnsupportedNvccFlag",
                f"Unsupported -Xcompiler host flag: {flag}",
                flag,
            )
    return flags


def _append_link_option(invocation, flag, value, cwd):
    if flag == "-L":
        path = Path(value)
        invocation.library_paths.append(str((cwd / path).resolve() if not path.is_absolute() else path.resolve()))
        return
    if not re.match(r"^[A-Za-z0-9_+.-]+$", value):
        raise NvccInvocationError("UnsupportedNvccFlag", f"Invalid library name: {value}", value)
    invocation.libraries.append(value)
