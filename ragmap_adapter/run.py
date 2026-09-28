"""``ragmap-run``: build Embodied-RAG's semantic forest from a RAGMAP object map.

    ragmap-run --input /input --output /output [SECTION.key=value ...]

Runs the release's own offline build -- ``generate_semantic_forest.py``'s
``generate_semantic_forest``, which drives
``embodied_nav.spatial_relationship_extractor.SpatialRelationshipExtractor``
and ``embodied_nav.llm.LLMInterface`` -- unmodified, GML in and GML out, and
converts the forest's cluster nodes into RAGMAP's area contract.

INPUT (``/input``, read-only)::

    meta.json      {"up_axis", "frame", "units": "metre", ...}
    objects.jsonl  {"id", "label", "centroid", ...} per leaf object

``centroid`` is in metres, in a frame whose vertical axis is ``up_axis``.

OUTPUT (``/output``)::

    areas.jsonl    {"id", "label", "summary", "centroid", "level", "parent",
                    "children", "extra"} per area, same frame and units as the input
    run.json       status, counts (areas per level, fallback names, ...), timings
    llm_calls.jsonl  every LLM request the build made: prompt, reply, error
    direct_semantic_graph_ragmap.gml / enhanced_semantic_graph_ragmap.gml
                   upstream's own input and output graphs, kept verbatim

``children`` are area ids from ``areas.jsonl`` and/or leaf ids from the
input ``objects.jsonl``; ``parent`` is an area id or null (a root).

What the adapter does, and all it does:

1. **Writes the input graph the way upstream's ground-truth logger does**
   (``embodied_nav/direct_scene_logger.py``, the AirSim object dump the paper's
   forests were built from): one node per object, keyed by a unique instance
   name, with ``position {x, y, z}``, ``type 'object'``, ``level 1``, ``name``,
   ``label`` and ``summary "Object: <name>"``. The instance name is
   ``<label>_<n>`` (a per-label counter), standing in for AirSim's unique
   object names. The ``summary`` matters: ``LLMInterface.generate_community_summary``
   describes a member *with* a summary as ``Area: <name> / Summary: <summary>``,
   which is what upstream's LLM saw for every leaf.
2. **Frames.** Upstream is z-up (the logger negates AirSim's NED z) and names
   directions by ``Config.CARDINAL_DIRECTIONS``: north ``+y``, east ``+x``.
   The input's declared ``up_axis`` is rotated onto ``+z`` before the graph is
   written and every returned position is rotated back. For ``up_axis: z``
   (every RAGMAP build) that is the identity, and "north" is simply the map's
   ``+y``, which is as arbitrary as upstream's own AirSim-axis naming.
3. **Configuration only.** ``Config.LLM['vllm_settings']`` is pointed at the
   OpenAI-compatible endpoint in ``RAGMAP_VLM_BASE_URL`` / ``RAGMAP_VLM_MODEL``,
   and ``SECTION.key=value`` arguments set entries of upstream's ``Config``
   dicts (``LLM.temperature=0.0``, ``SPATIAL.spatial_threshold=3.0``, ...).
4. **A recording subclass of LLMInterface** that logs each call and changes
   nothing about it.
5. **Refuses a silently truncated forest.** Upstream catches any exception in
   a level (an LLM request that failed included), prints "Clustering failed at
   level", and returns whatever it had -- a forest whose top level has no
   names. That is reported here as a failed run instead of an area map.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
import os
import platform
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger("ragmap_adapter")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ragmap.area_mapping/v1"
INPUT_GML = "direct_semantic_graph_ragmap.gml"
#: Upstream's ``__main__`` derives its output name exactly this way.
OUTPUT_GML = INPUT_GML.replace("direct_semantic_graph", "enhanced_semantic_graph")
#: ``LLMInterface.generate_community_summary``'s fallback, verbatim.
FALLBACK_NAME = "undefined_zone"
FALLBACK_SUMMARY = "Area containing multiple objects or spaces"

#: Rotation taking the declared up axis onto upstream's +z (row-major 3x3).
_TO_UPSTREAM: dict[str, tuple[tuple[float, float, float], ...]] = {
    "z": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    # +90 degrees about x: +y -> +z, +z -> -y.
    "y": ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ragmap-run", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "overrides", nargs="*",
        help="Entries of upstream's Config dicts, e.g. LLM.temperature=0.0 or LLM.vllm_settings.model=x",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def _rotate(matrix, vector):
    return tuple(sum(row[i] * vector[i] for i in range(3)) for row in matrix)


def _transpose(matrix):
    return tuple(tuple(matrix[j][i] for j in range(3)) for i in range(3))


def _rotation(up_axis: str):
    try:
        return _TO_UPSTREAM[up_axis]
    except KeyError:
        raise ValueError(f"meta.json up_axis must be one of {sorted(_TO_UPSTREAM)}, got {up_axis!r}") from None


# ---------------------------------------------------------------------------
# Upstream configuration
# ---------------------------------------------------------------------------


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def apply_overrides(config_cls: Any, overrides: list[str]) -> dict[str, Any]:
    """Set ``SECTION.key[.key...]=value`` entries of upstream's ``Config`` dicts.

    Only existing keys: a typo must not silently configure nothing.
    """

    applied: dict[str, Any] = {}
    for item in overrides:
        path, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override {item!r} is not SECTION.key=value")
        section, *keys = path.split(".")
        target = getattr(config_cls, section, None)
        if not isinstance(target, dict) or not keys:
            raise ValueError(f"override {item!r}: Config.{section} is not a dict section")
        for key in keys[:-1]:
            if not isinstance(target.get(key), dict):
                raise ValueError(f"override {item!r}: Config.{section} has no dict {key!r}")
            target = target[key]
        if keys[-1] not in target:
            raise ValueError(f"override {item!r}: Config.{path.rsplit('.', 1)[0]} has no key {keys[-1]!r}")
        value = _parse_value(raw)
        target[keys[-1]] = value
        applied[path] = value
    return applied


def point_llm_at_ragmap(config_cls: Any) -> dict[str, Any]:
    """Aim ``Config.LLM['vllm_settings']`` at the RAGMAP VLM service; hosted APIs never."""

    base_url = (os.environ.get("RAGMAP_VLM_BASE_URL") or "").strip().rstrip("/")
    model = (os.environ.get("RAGMAP_VLM_MODEL") or "").strip()
    if not base_url or not model:
        raise SystemExit(
            "RAGMAP_VLM_BASE_URL and RAGMAP_VLM_MODEL must name an OpenAI-compatible local "
            "server (RAGMAP's ragmap-vlm-service). The hosted OpenAI API is never used."
        )
    # `LLMInterface` appends "/v1" itself.
    root = base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url
    settings = config_cls.LLM["vllm_settings"]
    settings.update(enabled=True, model=model, api_base=root, api_key=os.environ.get("RAGMAP_VLM_API_KEY", "ragmap"))
    return {"api_base": root, "model": model}


# ---------------------------------------------------------------------------
# LLM call recording (subclass; behaviour unchanged)
# ---------------------------------------------------------------------------


def recording_llm_interface(base: type, calls: list[dict[str, Any]]) -> type:
    class RecordingLLMInterface(base):  # type: ignore[misc, valid-type]
        """Upstream's ``LLMInterface``, recording each ``generate_response`` call."""

        async def generate_response(self, prompt, system_prompt=None):
            record: dict[str, Any] = {"prompt": prompt, "system_prompt": system_prompt}
            started = time.monotonic()
            try:
                reply = await super().generate_response(prompt, system_prompt)
            except Exception as exc:
                record.update(error=f"{type(exc).__name__}: {exc}", seconds=round(time.monotonic() - started, 3))
                calls.append(record)
                raise
            record.update(reply=reply, seconds=round(time.monotonic() - started, 3))
            calls.append(record)
            return reply

        async def generate_community_summary(self, objects):
            result = await super().generate_community_summary(objects)
            if calls:
                calls[-1]["parsed"] = dict(result)
            return result

    return RecordingLLMInterface


class _Tee(io.TextIOBase):
    """Stdout that also remembers upstream's "Clustering failed" line."""

    def __init__(self, stream):
        self.stream = stream
        self.failures: list[str] = []
        self._pending = ""

    def write(self, text):
        self.stream.write(text)
        self._pending += text
        *lines, self._pending = self._pending.split("\n")
        self.failures.extend(line for line in lines if line.startswith("Clustering failed at level"))
        return len(text)

    def flush(self):
        self.stream.flush()


# ---------------------------------------------------------------------------
# INPUT -> upstream graph
# ---------------------------------------------------------------------------


def read_leaves(input_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = json.loads((input_dir / "meta.json").read_text(encoding="utf-8"))
    leaves: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate((input_dir / "objects.jsonl").read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        leaf_id = str(value.get("id") or "")
        label = str(value.get("label") or "").strip()
        centroid = value.get("centroid")
        if not leaf_id or leaf_id in seen:
            raise ValueError(f"objects.jsonl:{line_no}: missing or duplicate id {leaf_id!r}")
        if not label:
            raise ValueError(f"objects.jsonl:{line_no}: {leaf_id} has no label")
        if not isinstance(centroid, list) or len(centroid) != 3 or not all(
            isinstance(item, (int, float)) and math.isfinite(item) for item in centroid
        ):
            raise ValueError(f"objects.jsonl:{line_no}: {leaf_id} centroid must be three finite numbers")
        seen.add(leaf_id)
        leaves.append({"id": leaf_id, "label": label, "centroid": [float(item) for item in centroid]})
    return meta, leaves


def instance_names(leaves: list[dict[str, Any]]) -> list[str]:
    """``<label>_<n>`` per leaf, unique, standing in for AirSim's object names."""

    counters: Counter[str] = Counter()
    used: set[str] = set()
    names: list[str] = []
    for leaf in leaves:
        while True:
            counters[leaf["label"]] += 1
            name = f"{leaf['label']}_{counters[leaf['label']]}"
            if name not in used:
                break
        used.add(name)
        names.append(name)
    return names


def write_input_graph(path: Path, leaves: list[dict[str, Any]], names: list[str], rotation) -> None:
    """The graph ``DirectSceneLogger.build_topological_graph`` + ``save_graph`` writes."""

    import networkx as nx

    graph = nx.Graph()
    for leaf, name in zip(leaves, names):
        x, y, z = _rotate(rotation, leaf["centroid"])
        graph.add_node(
            name,
            position={"x": float(x), "y": float(y), "z": float(z)},
            type="object",
            level=1,
            name=name,
            label=name,
            summary=f"Object: {name}",
        )
    graph.graph["environment"] = "ragmap"
    graph.graph["timestamp"] = time.strftime("%Y%m%d_%H%M%S")
    graph.graph["creation_method"] = "ragmap_adapter (direct_scene_logger schema)"
    nx.write_gml(graph, path)


# ---------------------------------------------------------------------------
# upstream graph -> OUTPUT
# ---------------------------------------------------------------------------


def _position(data: dict[str, Any]) -> tuple[float, float, float] | None:
    value = data.get("position")
    if isinstance(value, dict) and all(key in value for key in ("x", "y", "z")):
        return (float(value["x"]), float(value["y"]), float(value["z"]))
    return None


def convert_forest(path: Path, name_to_leaf: dict[str, str], rotation) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cluster nodes of the enhanced GML as area records, plus counts."""

    import networkx as nx

    graph = nx.read_gml(path)
    back = _transpose(rotation)
    clusters = {node: data for node, data in graph.nodes(data=True) if data.get("type") == "cluster"}
    parent: dict[str, str] = {}
    children: dict[str, list[str]] = {node: [] for node in clusters}
    spatial: dict[str, list[dict[str, Any]]] = {node: [] for node in clusters}
    spatial_by_level: Counter[str] = Counter()
    for u, v, data in graph.edges(data=True):
        if data.get("relationship") == "part_of":
            up, down = (u, v) if u in clusters and (v not in clusters or clusters[u]["level"] > clusters[v]["level"]) else (v, u)
            if up not in clusters:
                raise ValueError(f"part_of edge {u!r} - {v!r} joins two non-cluster nodes")
            if down in parent and parent[down] != up:
                raise ValueError(f"{down!r} is part_of both {parent[down]!r} and {up!r}")
            parent[down] = up
            children[up].append(down)
        elif data.get("type") == "spatial":
            level = graph.nodes[u].get("level")
            spatial_by_level[str(level)] += 1
            edge = {
                # `_get_cardinal_direction(source, target)`: the target lies
                # `relationship` of the source. Edges keep upstream's
                # (node1, node2) order through the merge and the GML round trip.
                "source": u, "target": v, "relationship": data.get("relationship"),
                "distance_m": float(data.get("distance") or 0.0),
            }
            for end in (u, v):
                if end in spatial:
                    spatial[end].append(edge)

    def contract_id(node: str) -> str:
        if node in clusters:
            return node
        try:
            return name_to_leaf[node]
        except KeyError:
            raise ValueError(f"forest node {node!r} is neither an area nor an input leaf") from None

    def leaf_count(node: str) -> int:
        return 1 if node not in clusters else sum(leaf_count(child) for child in children[node])

    areas: list[dict[str, Any]] = []
    missing_names: list[str] = []
    for node, data in clusters.items():
        position = _position(data)
        if position is None:
            raise ValueError(f"area {node!r} has no position")
        name, summary = data.get("name"), data.get("summary")
        if not name or summary is None:
            missing_names.append(node)
        areas.append({
            "id": node,
            "label": str(name or ""),
            "summary": str(summary or ""),
            "centroid": list(_rotate(back, position)),
            "level": int(data["level"]),
            "parent": parent.get(node),
            "children": [contract_id(child) for child in children[node]],
            "extra": {
                "upstream_id": node,
                "direct_children": len(children[node]),
                "leaves": leaf_count(node),
                "spatial_edges": spatial[node],
                "name_fallback": name == FALLBACK_NAME,
            },
        })
    areas.sort(key=lambda item: (item["level"], item["id"]))
    leaf_nodes = [node for node, data in graph.nodes(data=True) if data.get("type") != "cluster"]
    object_spatial = sum(
        1 for u, v, data in graph.edges(data=True)
        if data.get("type") == "spatial" and u not in clusters and v not in clusters
    )
    counts = {
        "areas": len(areas),
        "levels": dict(sorted(Counter(str(area["level"]) for area in areas).items())),
        "roots": sum(1 for area in areas if area["parent"] is None),
        "leaves_in_forest": sum(1 for node in leaf_nodes if node in parent),
        "leaves_outside_forest": sum(1 for node in leaf_nodes if node not in parent),
        "spatial_edges_by_level": dict(sorted(spatial_by_level.items())),
        "object_spatial_edges": object_spatial,
        "fallback_names": sum(1 for area in areas if area["label"] == FALLBACK_NAME),
        "fallback_unparsed": sum(
            1 for area in areas if area["label"] == FALLBACK_NAME and area["summary"] == FALLBACK_SUMMARY
        ),
        "missing_names": missing_names,
    }
    return areas, counts


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _versions() -> dict[str, str]:
    import networkx
    import numpy
    import openai
    import scipy
    import sklearn

    return {
        "python": platform.python_version(), "networkx": networkx.__version__, "numpy": numpy.__version__,
        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__, "openai": openai.__version__,
    }


def run(input_dir: Path, output_dir: Path, overrides: list[str], report: dict[str, Any]) -> None:
    """Fill *report* as the run goes, so a failure's run.json keeps what was learnt."""

    sys.path.insert(0, str(REPO_ROOT))
    import generate_semantic_forest as gsf
    from embodied_nav.config import Config

    output_dir.mkdir(parents=True, exist_ok=True)
    report.update({
        "schema": SCHEMA, "status": "running", "input": str(input_dir), "output": str(output_dir),
        "embodied_rag_git_sha": os.environ.get("EMBODIED_RAG_GIT_SHA"), "versions": _versions(),
    })
    timings: dict[str, float] = {}
    report["timings"] = timings
    meta, leaves = read_leaves(input_dir)
    up_axis = str(meta.get("up_axis", "z"))
    rotation = _rotation(up_axis)
    report["input_meta"] = meta
    report["up_axis"] = {"declared": up_axis, "to_upstream": [list(row) for row in rotation]}

    endpoint = point_llm_at_ragmap(Config)
    report["overrides"] = apply_overrides(Config, overrides)
    report["llm"] = {
        **endpoint,
        "temperature": Config.LLM["temperature"], "max_tokens": Config.LLM["max_tokens"],
    }
    report["spatial_config"] = dict(Config.SPATIAL)

    names = instance_names(leaves)
    name_to_leaf = dict(zip(names, (leaf["id"] for leaf in leaves)))
    # Upstream drops these without saying so; counted here so it is never a surprise.
    report["counts"] = {
        "leaves_in": len(leaves),
        "dropped_at_origin": sum(1 for leaf in leaves if all(item == 0 for item in _rotate(rotation, leaf["centroid"]))),
        "dropped_drone_named": sum(1 for name in names if "drone" in name.lower()),
    }
    input_gml, output_gml = output_dir / INPUT_GML, output_dir / OUTPUT_GML
    write_input_graph(input_gml, leaves, names, rotation)

    calls: list[dict[str, Any]] = []
    gsf.LLMInterface = recording_llm_interface(gsf.LLMInterface, calls)
    tee = _Tee(sys.stdout)
    started = time.monotonic()
    real_stdout, sys.stdout = sys.stdout, tee
    try:
        asyncio.run(gsf.generate_semantic_forest(str(input_gml), str(output_gml)))
    finally:
        sys.stdout = real_stdout
        timings["forest_seconds"] = round(time.monotonic() - started, 3)
        with (output_dir / "llm_calls.jsonl").open("w", encoding="utf-8") as handle:
            for call in calls:
                handle.write(json.dumps(call) + "\n")

    areas, forest_counts = convert_forest(output_gml, name_to_leaf, rotation)
    errors = [call["error"] for call in calls if "error" in call]
    report["counts"].update(forest_counts)
    report["counts"]["llm_calls"] = len(calls)
    report["counts"]["llm_errors"] = len(errors)
    if tee.failures or errors or forest_counts["missing_names"]:
        raise RuntimeError(
            "Embodied-RAG's forest builder stopped early and would have returned a truncated forest: "
            f"{'; '.join(tee.failures) or 'no clustering failure printed'}; "
            f"{len(errors)} LLM error(s){': ' + errors[0] if errors else ''}; "
            f"{len(forest_counts['missing_names'])} area(s) with no name."
        )
    with (output_dir / "areas.jsonl").open("w", encoding="utf-8") as handle:
        for area in areas:
            handle.write(json.dumps(area) + "\n")
    report["status"] = "ok"


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    started = time.monotonic()
    report: dict[str, Any] = {"schema": SCHEMA}
    try:
        run(args.input, args.output, list(args.overrides), report)
        code = 0
    except SystemExit:
        raise
    except Exception as exc:  # recorded for the RAGMAP side, which reads run.json first
        logger.exception("ragmap-run failed")
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        code = 1
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    counts = report.get("counts") or {}
    logger.info(
        "status=%s areas=%s levels=%s fallback_names=%s llm_calls=%s -> %s",
        report["status"], counts.get("areas"), counts.get("levels"), counts.get("fallback_names"),
        counts.get("llm_calls"), args.output,
    )
    return code


if __name__ == "__main__":
    sys.exit(main())
