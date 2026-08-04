"""MedSwin runtime core. Import MedSwinOrchestrator from app.medswin.orchestrator."""

__all__ = ["MedSwinOrchestrator"]


def __getattr__(name: str):
    if name == "MedSwinOrchestrator":
        from app.medswin.orchestrator import MedSwinOrchestrator

        return MedSwinOrchestrator
    raise AttributeError(name)
