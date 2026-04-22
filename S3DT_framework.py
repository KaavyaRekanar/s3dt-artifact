Install dependencies
"""

!pip -q install networkx pydantic numpy
# Optional for embeddings in Algorithm 3:
!pip -q install sentence-transformers

"""Core data model (claims, deltas, constraints, events)"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Callable, Iterable
from pydantic import BaseModel, Field
import networkx as nx
import numpy as np
from dataclasses import dataclass

# ----------------------------
# Canonical claim model
# ----------------------------
class Claim(BaseModel):
    s: str
    p: str
    o: str
    q: Dict[str, Any] = Field(default_factory=dict)   # qualifiers/scope
    gamma: float = 1.0                                # extraction confidence [0,1]
    evidence: List[str] = Field(default_factory=list) # evidence pointers/snippets

    def key(self) -> Tuple[str, str, Tuple[Tuple[str, Any], ...]]:
        # Canonical key for alignment: subject + predicate + qualifier scope (sorted)
        return (self.s, self.p, tuple(sorted(self.q.items())))

class ClaimSet(BaseModel):
    claims: List[Claim] = Field(default_factory=list)

    def by_key(self) -> Dict[Tuple, Claim]:
        return {c.key(): c for c in self.claims}

# ----------------------------
# Delta model (edge annotation)
# ----------------------------
class ClaimDelta(BaseModel):
    added: List[Claim] = Field(default_factory=list)
    removed: List[Claim] = Field(default_factory=list)
    modified: List[Tuple[Claim, Claim]] = Field(default_factory=list)  # (old,new)

    provenance: Dict[str, Any] = Field(default_factory=dict)

# --------------------------------------------
# Semantic Integrity Constraint (SIC) model
# ---------------------------------------------
class SIC(BaseModel):
    id: str
    description: str
    # scope: which predicates/subjects matter for this constraint
    predicates: Optional[List[str]] = None
    subjects: Optional[List[str]] = None

    epsilon: float = 0.2      # tolerance threshold
    lambda_p: float = 0.7     # fusion weight structured vs embedding

    # optional severity/criticality weight (useful inside structured scoring)
    w_p: float = 1.0

# ----------------------------
# Candidate + event model
# ----------------------------
class DriftCandidate(BaseModel):
    source: str
    target: str
    constraint_id: str
    result: str   # "VIOLATION" or "SUSPECT"
    delta: ClaimDelta

class DriftEvent(BaseModel):
    source: str
    target: str
    constraint_id: str
    severity: float
    drift_type: str
    explanation: str

"""Helper functions for Algorithm 1 (projection + delta)"""

# ----------------------------
# Artifact normalization (stub)
# ----------------------------
def normalize_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """
    In practice: parse file/config/spec/log into canonical internal form.
    Here: treat artifact dict as already normalized.
    """
    return artifact

# ----------------------------
# Semantic claims extraction (stub)
# ----------------------------
def semantic_claims(v_norm: Dict[str, Any]) -> ClaimSet:
    """
    In practice: extract claims from text/config/code/etc.
    Here: assume v_norm already contains 'claims' in canonical form.
    """
    claims = [Claim(**c) for c in v_norm.get("claims", [])]
    return ClaimSet(claims=claims)

# ----------------------------
# Align + diff between claim sets
# ----------------------------
def align_and_diff(Ci: ClaimSet, Cj: ClaimSet) -> ClaimDelta:
    mi, mj = Ci.by_key(), Cj.by_key()
    keys_i, keys_j = set(mi.keys()), set(mj.keys())

    added = [mj[k] for k in sorted(keys_j - keys_i)]
    removed = [mi[k] for k in sorted(keys_i - keys_j)]

    modified = []
    for k in sorted(keys_i & keys_j):
        old, new = mi[k], mj[k]
        # value change if object differs
        if old.o != new.o:
            modified.append((old, new))

    return ClaimDelta(added=added, removed=removed, modified=modified)

def attach_provenance(delta: ClaimDelta, e_ij: Dict[str, Any]) -> ClaimDelta:
    delta.provenance = dict(e_ij)
    return delta

"""Digital thread graph store"""

def update_graph_node(G: nx.DiGraph, node_id: str, claims: ClaimSet, meta: Optional[Dict[str, Any]] = None):
    G.add_node(node_id)
    G.nodes[node_id]["claims"] = claims
    if meta:
        G.nodes[node_id].update(meta)

def update_graph_edge(G: nx.DiGraph, src: str, dst: str, delta: ClaimDelta, meta: Optional[Dict[str, Any]] = None):
    G.add_edge(src, dst)
    G.edges[src, dst]["delta"] = delta
    if meta:
        G.edges[src, dst].update(meta)

def get_claims(G: nx.DiGraph, node_id: str) -> ClaimSet:
    return G.nodes[node_id]["claims"]

def get_edge_delta(G: nx.DiGraph, src: str, dst: str) -> ClaimDelta:
    return G.edges[src, dst]["delta"]

"""Algorithm 1: ProjectAndAnnotateThreadEdge"""

def ProjectAndAnnotateThreadEdge(
    G: nx.DiGraph,
    a_i: Dict[str, Any],
    a_j: Dict[str, Any],
    e_ij: Dict[str, Any],
    node_i: str,
    node_j: str
) -> Tuple[str, str, ClaimDelta]:
    """
    Algorithm 1: Semantic Projection Engine
    - normalize artifacts
    - extract canonical semantic claims
    - compute meaning-level delta on the edge
    - persist to graph
    """
    v_i = normalize_artifact(a_i)
    v_j = normalize_artifact(a_j)

    C_i = semantic_claims(v_i)
    C_j = semantic_claims(v_j)

    delta_ij = align_and_diff(C_i, C_j)
    delta_ij = attach_provenance(delta_ij, e_ij)

    update_graph_node(G, node_i, C_i, meta={"artifact_meta": v_i.get("meta", {})})
    update_graph_node(G, node_j, C_j, meta={"artifact_meta": v_j.get("meta", {})})
    update_graph_edge(G, node_i, node_j, delta_ij, meta={"link_meta": e_ij})

    return node_i, node_j, delta_ij

"""Algorithm 2: TraverseThreadAndDetectDrift (SIC evaluation)"""

def select_claims(C: ClaimSet, p: SIC) -> ClaimSet:
    out = []
    for c in C.claims:
        if p.predicates and c.p not in p.predicates:
            continue
        if p.subjects and c.s not in p.subjects:
            continue
        out.append(c)
    return ClaimSet(claims=out)

def restrict_delta_to_scope(delta: ClaimDelta, p: SIC) -> ClaimDelta:
    def in_scope(c: Claim) -> bool:
        if p.predicates and c.p not in p.predicates:
            return False
        if p.subjects and c.s not in p.subjects:
            return False
        return True

    scoped = ClaimDelta(
        added=[c for c in delta.added if in_scope(c)],
        removed=[c for c in delta.removed if in_scope(c)],
        modified=[(o,n) for (o,n) in delta.modified if in_scope(o) or in_scope(n)],
        provenance=dict(delta.provenance),
    )
    return scoped

def evaluate_constraint(p: SIC, Ci: ClaimSet, Cj: ClaimSet, delta_ij: ClaimDelta) -> str:
    """
    Returns: "SATISFIED", "SUSPECT", "VIOLATION"
    Minimal placeholder logic:
      - if no scoped changes => SATISFIED
      - if scoped changes exist => SUSPECT
      - if change removes or flips a critical predicate (example) => VIOLATION
    """
    scoped = restrict_delta_to_scope(delta_ij, p)
    any_change = bool(scoped.added or scoped.removed or scoped.modified)
    if not any_change:
        return "SATISFIED"

    # Example "hard violation" heuristic: removal of a scoped claim with predicate 'encryption_at_rest'
    for c in scoped.removed:
        if c.p == "encryption_at_rest":
            return "VIOLATION"

    return "SUSPECT"

def TraverseThreadAndDetectDrift(G: nx.DiGraph, P: List[SIC]) -> List[DriftEvent]:
    """
    Algorithm 2: Drift Detection Engine
    - traverse graph edges
    - evaluate SICs
    - emit drift candidates and immediately call Algorithm 3 for scoring/classification
    """
    D: List[DriftEvent] = []
    for (vi, vj) in G.edges():
        Ci = get_claims(G, vi)
        Cj = get_claims(G, vj)
        delta_ij = get_edge_delta(G, vi, vj)

        for p in P:
            # applicability check (simple): if scope overlaps either node claim set
            Ci_p = select_claims(Ci, p)
            Cj_p = select_claims(Cj, p)
            if not (Ci_p.claims or Cj_p.claims):
                continue

            result = evaluate_constraint(p, Ci, Cj, delta_ij)
            if result in ("VIOLATION", "SUSPECT"):
                candidate = DriftCandidate(
                    source=vi,
                    target=vj,
                    constraint_id=p.id,
                    result=result,
                    delta=delta_ij
                )
                event = ScoreClassifyExplain(G, candidate, p)
                D.append(event)

    return D

"""Algorithm 3: ScoreClassifyExplain (hybrid scoring + explanation)"""

from sentence_transformers import SentenceTransformer

# swap this model later; it's small-ish and common for demos.
_EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")

def risk_weighted_structured_delta(scoped_delta: ClaimDelta, p: SIC) -> float:
    """
    Simple auditable structured score:
      weighted sum of (#added + #removed + #modified), scaled by constraint weight.
    Replace with your w_c, w_t, gamma weighting logic later.
    """
    count = len(scoped_delta.added) + len(scoped_delta.removed) + len(scoped_delta.modified)
    # cap to keep stable; normalization happens later
    return p.w_p * float(count)

def normalize_structured(d: float, p: SIC) -> float:
    """
    Normalize to [0,1]. Here: saturating normalization.
    """
    return float(1.0 - np.exp(-d / 5.0))  # tweak denominator for sensitivity

def render_summary(Cp: ClaimSet) -> str:
    """
    Canonical summary of scoped claims.
    """
    lines = []
    for c in Cp.claims:
        q = ",".join([f"{k}={v}" for k,v in sorted(c.q.items())])
        lines.append(f"{c.s} | {c.p} | {c.o} | {q}")
    return "\n".join(lines) if lines else "<empty>"

def calibrate_embed(d_embed: float, p: SIC) -> float:
    """
    Placeholder calibration: clamp to [0,1].
    You can replace with constraint-specific baselines later.
    """
    return float(np.clip(d_embed, 0.0, 1.0))

def classify_drift(p: SIC, result: str, drift_score: float, scoped_delta: ClaimDelta) -> str:
    """
    Simple security-theoretic classification:
      - below tolerance => benign
      - violation + high score => dangerous/adversarial
    """
    if drift_score <= p.epsilon:
        return "benign"
    if result == "VIOLATION" and drift_score > max(0.8, p.epsilon + 0.3):
        return "adversarial"
    if result == "VIOLATION":
        return "dangerous"
    return "instability"  # SUSPECT above tolerance

def generate_explanation(p: SIC, scoped_delta: ClaimDelta) -> str:
    """
    Human-auditable explanation grounded in explicit claim changes.
    """
    parts = [f"SIC={p.id}: {p.description}"]
    if scoped_delta.added:
        parts.append(f"Added ({len(scoped_delta.added)}): " + "; ".join([f"{c.s}:{c.p}={c.o}" for c in scoped_delta.added[:5]]))
    if scoped_delta.removed:
        parts.append(f"Removed ({len(scoped_delta.removed)}): " + "; ".join([f"{c.s}:{c.p}={c.o}" for c in scoped_delta.removed[:5]]))
    if scoped_delta.modified:
        parts.append(f"Modified ({len(scoped_delta.modified)}): " + "; ".join([f"{o.s}:{o.p}:{o.o}→{n.o}" for (o,n) in scoped_delta.modified[:5]]))
    if scoped_delta.provenance:
        parts.append(f"Provenance: {scoped_delta.provenance}")
    return " | ".join(parts)

def ScoreClassifyExplain(G: nx.DiGraph, candidate: DriftCandidate, p: SIC) -> DriftEvent:
    """
    Algorithm 3: hybrid scoring + classification + explanation.
    """
    vi, vj = candidate.source, candidate.target
    Ci = get_claims(G, vi)
    Cj = get_claims(G, vj)

    # policy scoped selection
    Ci_p = select_claims(Ci, p)
    Cj_p = select_claims(Cj, p)

    # structured component from scoped delta
    scoped_delta = restrict_delta_to_scope(candidate.delta, p)
    D_structured = risk_weighted_structured_delta(scoped_delta, p)
    D_structured_hat = normalize_structured(D_structured, p)

    # embedding component from summaries
    Ti = render_summary(Ci_p)
    Tj = render_summary(Cj_p)
    ei = _EMBEDDER.encode([Ti], normalize_embeddings=True)[0]
    ej = _EMBEDDER.encode([Tj], normalize_embeddings=True)[0]
    d_embed = float(1.0 - np.dot(ei, ej))
    D_embed_hat = calibrate_embed(d_embed, p)

    # fuse
    drift_score = float(p.lambda_p * D_structured_hat + (1.0 - p.lambda_p) * D_embed_hat)

    # classify + explain
    drift_type = classify_drift(p, candidate.result, drift_score, scoped_delta)
    explanation = generate_explanation(p, scoped_delta)

    return DriftEvent(
        source=vi,
        target=vj,
        constraint_id=p.id,
        severity=drift_score,
        drift_type=drift_type,
        explanation=explanation
    )

"""Demo: build a tiny thread and run the pipeline"""

# Build a demo digital thread
G = nx.DiGraph()

artifact_A = {
    "meta": {"name": "Spec A", "version": "1.0"},
    "claims": [
        {"s": "StorageBucketX", "p": "encryption_at_rest", "o": "AES-256", "q": {"region":"EU"}, "gamma": 0.95, "evidence": ["spec:A:12"]},
        {"s": "DatasetA", "p": "contains", "o": "PII", "q": {"region":"EU"}, "gamma": 0.9, "evidence": ["spec:A:20"]},
    ]
}
artifact_B = {
    "meta": {"name": "Config B", "version": "1.1"},
    "claims": [
        # encryption claim removed => likely violation
        {"s": "DatasetA", "p": "contains", "o": "PII", "q": {"region":"EU"}, "gamma": 0.9, "evidence": ["config:B:33"]},
    ]
}
link_meta = {"type": "derived-from", "commit": "abc123", "timestamp": "2026-02-15"}

# Algorithm 1: project + annotate edge
ProjectAndAnnotateThreadEdge(G, artifact_A, artifact_B, link_meta, node_i="ArtifactA", node_j="ArtifactB")

# Define constraints
P = [
    SIC(
        id="SIC-ENC-001",
        description="All EU storage must enforce encryption at rest.",
        predicates=["encryption_at_rest"],
        epsilon=0.2,
        lambda_p=0.8,
        w_p=1.5
    ),
    SIC(
        id="SIC-PII-001",
        description="PII handling claims must remain consistent across transformations.",
        predicates=["contains"],
        subjects=["DatasetA"],
        epsilon=0.25,
        lambda_p=0.6,
        w_p=1.0
    )
]

# Algorithm 2 (calls Algorithm 3)
events = TraverseThreadAndDetectDrift(G, P)

events

"""Printing Results"""

for e in events:
    print(f"[{e.drift_type.upper()}] {e.constraint_id} {e.source} → {e.target} | severity={e.severity:.3f}")
    print(f"  {e.explanation}\n")
