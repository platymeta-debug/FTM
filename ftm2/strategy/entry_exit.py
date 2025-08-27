from ftm2.config.settings import load_env_chain

CFG = load_env_chain()


def entry_signal(score: float) -> str | None:
    """Return 'long', 'short', or None based on score thresholds."""
    if score > 60:
        return "long"
    if score < 40:
        return "short"
    return None


def exit_signal(score: float) -> bool:
    """Return True if position should be exited."""
    return score < 50
