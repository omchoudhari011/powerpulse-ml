"""Project-specific exceptions so the CLIs can print clear messages."""


class PowerPulseError(Exception):
    """Base class for all PowerPulse ML errors."""


class ConfigError(PowerPulseError, ValueError):
    """Invalid configuration value."""


class DatasetError(PowerPulseError, ValueError):
    """Dataset cannot be loaded or fails validation."""


class InsufficientDataError(DatasetError):
    """Not enough (valid) data to train, evaluate or forecast reliably."""


class ArtifactError(PowerPulseError):
    """Trained artifacts are missing, incomplete or incompatible."""


class RAGError(PowerPulseError):
    """The question-answering (RAG) layer cannot index documents or get an answer."""
