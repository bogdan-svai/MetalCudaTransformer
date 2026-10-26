"""Evaluate declarative host-loop guards. Not wired to GPU dispatch yet."""


def evaluate_guards(guards, captures):
    """Unknown/missing/invalid evidence always keeps the CPU path."""
    for guard in guards:
        if guard.get("kind") != "unsigned_le_constant":
            return False
        value = captures.get(guard.get("symbol_usr"))
        maximum = guard.get("maximum")
        if type(value) is not int or type(maximum) is not int:
            return False
        if not 0 <= maximum <= 4294967295 or not 0 <= value <= maximum:
            return False
    return True
