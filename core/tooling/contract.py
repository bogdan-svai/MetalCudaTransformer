from copy import deepcopy

from core.version import MCU_VERSION


CONTRACT_SCHEMA_VERSION = 2
JSON_CONTRACT = "mcu.tooling.v2"


def with_contract(payload, exit_code=None):
    result = deepcopy(payload)
    result.setdefault("schema_version", CONTRACT_SCHEMA_VERSION)
    result.setdefault("mcu_version", MCU_VERSION)
    result.setdefault("contract", JSON_CONTRACT)
    if exit_code is not None:
        result["exit_status"] = {
            "code": exit_code,
            "class": classify_exit_code(exit_code, result),
        }
    return result


def classify_exit_code(exit_code, payload=None):
    if exit_code == 0:
        return "success"
    if exit_code == 1:
        return "source_failure"
    if exit_code == 2:
        return "infrastructure_failure"
    if payload and payload.get("command") == "run":
        return "program_failure"
    return "program_failure"

