from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Literal

# =====================================================
# PROGRAMME DOCTORAT — NIVEAUX COGNITIFS (MÉTAMODÈLE)
# =====================================================

DoctoratLevel = Literal[
    "BAC",
    "LICENCE_1",
    "LICENCE_2",
    "LICENCE_3",
    "MASTER_1",
    "MASTER_2",
    "DOCTORAT_1",
    "DOCTORAT_2",
    "DOCTORAT_3",
    "POSTDOC_1",
    "POSTDOC_2",
    "POSTDOC_3",
    "EXPERT_1",
    "EXPERT_2",
    "EXPERT_3",
    "PRIX_NOBEL_1",
    "PRIX_NOBEL_2",
    "PRIX_NOBEL_3",
]


@dataclass
class DoctoratProfile:
    level: DoctoratLevel
    description: str
    # Plus le niveau est élevé, plus on est exigeant (score minimal acceptable)
    min_confidence: int  # 0-100
    max_allowed_risk: int  # 0-5 (R0-R5)


DOCTORAT_PROFILES: dict[DoctoratLevel, DoctoratProfile] = {
    "BAC": DoctoratProfile(
        level="BAC",
        description="Niveau d'entrée, expérimentation, tolérance élevée à l'incertitude.",
        min_confidence=40,
        max_allowed_risk=4,
    ),
    "MASTER_2": DoctoratProfile(
        level="MASTER_2",
        description="Production sérieuse, besoin de justification, mais sandbox encore acceptable.",
        min_confidence=60,
        max_allowed_risk=3,
    ),
    "DOCTORAT_1": DoctoratProfile(
        level="DOCTORAT_1",
        description="Entrée dans le protocole Z-DS, rigueur forte, contexte critique.",
        min_confidence=70,
        max_allowed_risk=2,
    ),
    "DOCTORAT_3": DoctoratProfile(
        level="DOCTORAT_3",
        description="Z-DS pleinement appliqué, missions sensibles.",
        min_confidence=80,
        max_allowed_risk=2,
    ),
    "PRIX_NOBEL_3": DoctoratProfile(
        level="PRIX_NOBEL_3",
        description="Exigence maximale, usage critique, tolérance quasi nulle à l'incertitude.",
        min_confidence=90,
        max_allowed_risk=1,
    ),
    # Les autres niveaux pourront être détaillés plus tard
}


def get_doctorat_profile(level: DoctoratLevel) -> DoctoratProfile:
    # Fallback simple : si un niveau n'est pas détaillé, on map sur MASTER_2
    return DOCTORAT_PROFILES.get(level) or DOCTORAT_PROFILES["MASTER_2"]


# =====================================================
# STRUCTURES DE SCORING RAG
# =====================================================

@dataclass
class RagScoringSignals:
    sources_count: int
    distinct_sources: int
    has_recent_metadata: bool
    has_multiple_sources: bool
    has_potential_conflicts: bool
    summary_length: int


@dataclass
class RagScoringResult:
    scored_at: str
    query: str
    confidence: int          # 0-100 (combinaison de sous-scores)
    source_quality: int      # 0-100
    coherence: int           # 0-100
    risk_level: int          # 0-5 (R0-R5)
    risk_label: str          # "R0"..."R5"
    signals: RagScoringSignals
    doctorat_profile: dict[str, Any]
    interpretation: str      # texte court pour la console
    recommendations: list[str]


# =====================================================
# FONCTIONS DE SCORING
# =====================================================

def _score_sources(chunks: list[dict[str, Any]]) -> tuple[int, RagScoringSignals]:
    """
    Very first heuristic:
    - plus il y a de sources distinctes, mieux c'est
    - on essaie de détecter une notion de "récent" si metadata.date existe
    - pas de détection fine de conflit pour l'instant (placeholder)
    """
    sources = [c.get("source") for c in chunks if c.get("source")]
    distinct_sources = len(set(sources))
    sources_count = len(sources)

    # Heuristique simple pour source_quality
    if distinct_sources == 0:
        source_quality = 20
    elif distinct_sources == 1:
        source_quality = 50
    elif distinct_sources <= 3:
        source_quality = 70
    else:
        source_quality = 85

    has_recent_metadata = False
    now = datetime.now(timezone.utc)

    for c in chunks:
        meta = c.get("metadata") or {}
        date_str = meta.get("date") or meta.get("created_at") or meta.get("updated_at")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            delta_days = (now - dt).total_seconds() / 86400
            if delta_days <= 30:
                has_recent_metadata = True
                break
        except Exception:
            continue

    has_multiple_sources = distinct_sources >= 2

    # Placeholder : plus tard tu pourras analyser les textes pour détecter de vraies contradictions.
    has_potential_conflicts = False

    signals = RagScoringSignals(
        sources_count=sources_count,
        distinct_sources=distinct_sources,
        has_recent_metadata=has_recent_metadata,
        has_multiple_sources=has_multiple_sources,
        has_potential_conflicts=has_potential_conflicts,
        summary_length=0,  # rempli plus tard
    )

    return source_quality, signals


def _score_coherence(summary: str, chunks: list[dict[str, Any]]) -> int:
    """
    Heuristique très simple pour commencer :
    - résumé très court ou très long -> suspicion
    - présence de citations ou de références simples -> +points
    Plus tard : scoring sémantique, cross-check, etc. [web:178][web:184]
    """
    length = len(summary)

    if length < 200:
        base = 50
    elif length < 1200:
        base = 80
    else:
        base = 70

    if "[" in summary and "]" in summary:
        base += 5

    return max(0, min(100, base))


def _derive_risk_level(
    confidence: int,
    source_quality: int,
    coherence: int,
) -> int:
    """
    Mappe les sous-scores vers un risque R0-R5.
    Règle simple : plus les scores sont bas, plus R est élevé. [web:181][web:182][web:188]
    """
    avg = (confidence + source_quality + coherence) / 3

    if avg >= 85:
        return 0
    if avg >= 70:
        return 1
    if avg >= 55:
        return 2
    if avg >= 40:
        return 3
    if avg >= 25:
        return 4
    return 5


def score_rag_answer(
    query: str,
    summary: str,
    chunks: list[dict[str, Any]],
    level: DoctoratLevel = "MASTER_2",
) -> dict[str, Any]:
    """
    Scoring RAG pour Zyqtron :
    - pensé comme un ENGINE (ne décide pas, n'agit pas),
    - compatible avec Z-DS 1.1 (R0-R5),
    - intègre ton programme doctorat comme profil d'exigence.
    """
    profile = get_doctorat_profile(level)

    source_quality, signals = _score_sources(chunks)
    coherence = _score_coherence(summary, chunks)

    # Pour l'instant, on fixe la "confidence" comme moyenne source+cohérence.
    confidence = int(round((source_quality + coherence) / 2))

    risk_level = _derive_risk_level(confidence, source_quality, coherence)
    risk_label = f"R{risk_level}"

    signals.summary_length = len(summary)

    recommendations: list[str] = []

    if confidence < profile.min_confidence:
        recommendations.append(
            f"Confiance {confidence} < seuil minimal {profile.min_confidence} pour le niveau {profile.level}."
        )

    if risk_level > profile.max_allowed_risk:
        recommendations.append(
            f"Risque {risk_label} supérieur au maximum recommandé (R{profile.max_allowed_risk}) pour {profile.level}."
        )

    if not signals.has_multiple_sources:
        recommendations.append("Réponse basée sur une seule source : vérifier manuellement ou compléter le RAG.")

    if not signals.has_recent_metadata:
        recommendations.append("Sources potentiellement anciennes : attention à la fraîcheur de l'information.")

    if signals.has_potential_conflicts:
        recommendations.append("Conflits potentiels détectés entre les sources : validation humaine recommandée.")

    if not recommendations:
        recommendations.append("Score cohérent avec le niveau Doctorat, aucune alerte forte.")

    interpretation = (
        f"Confiance={confidence}, Source={source_quality}, Cohérence={coherence}, "
        f"Risque={risk_label} pour niveau {profile.level}."
    )

    result = RagScoringResult(
        scored_at=datetime.now(timezone.utc).isoformat(),
        query=query,
        confidence=confidence,
        source_quality=source_quality,
        coherence=coherence,
        risk_level=risk_level,
        risk_label=risk_label,
        signals=signals,
        doctorat_profile=asdict(profile),
        interpretation=interpretation,
        recommendations=recommendations,
    )

    return asdict(result)
