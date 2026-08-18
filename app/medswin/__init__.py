"""MedSwin runtime core. Import MedSwinOrchestrator from app.medswin.orchestrator."""

__all__ = ["MedSwinOrchestrator", "NaiveRAGOrchestrator"]


def __getattr__(name: str):
    if name == "MedSwinOrchestrator":
        from app.medswin.orchestrator import MedSwinOrchestrator

        return MedSwinOrchestrator
    if name == "NaiveRAGOrchestrator":
        from app.medswin.naive import NaiveRAGOrchestrator

        return NaiveRAGOrchestrator
    raise AttributeError(name)
