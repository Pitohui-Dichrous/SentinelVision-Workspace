"""Class visibility and PPE evidence policy shared by UI integration tests."""

from __future__ import annotations

from typing import Mapping


def ppe_evidence_enabled(
    class_enabled: Mapping[str, bool],
    protected_class_id: str,
    unprotected_class_id: str,
) -> bool:
    """Keep full mutually-exclusive evidence when either PPE view is enabled."""

    return bool(
        class_enabled.get(protected_class_id, True)
        or class_enabled.get(unprotected_class_id, True)
    )


def stable_class_visible(
    class_id: str,
    class_enabled: Mapping[str, bool],
    protected_class_id: str,
    unprotected_class_id: str,
    uncertain_class_id: str,
) -> bool:
    """Apply display switches to the stable result, not to input evidence."""

    if class_id == uncertain_class_id:
        return ppe_evidence_enabled(
            class_enabled,
            protected_class_id,
            unprotected_class_id,
        )
    return bool(class_enabled.get(class_id, True))
