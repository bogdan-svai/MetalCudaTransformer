from dataclasses import dataclass


@dataclass(frozen=True)
class TranslationRule:
    source_symbol: str
    category: str
    supported_arities: tuple[int, ...]
    capability: str
    lowering_handler: str
    target_symbol: str | None = None

    def to_dict(self):
        return {
            "source_symbol": self.source_symbol,
            "category": self.category,
            "supported_arities": list(self.supported_arities),
            "capability": self.capability,
            "lowering_handler": self.lowering_handler,
            "target_symbol": self.target_symbol,
        }
