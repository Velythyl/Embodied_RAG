"""Write a small synthetic area-contract input: three rooms of objects, z up."""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rooms = {"kitchen": (0.0, 0.0), "office": (12.0, 0.0), "lounge": (0.0, 14.0)}
labels = {"kitchen": ["fridge", "sink", "microwave", "table"], "office": ["desk", "chair", "monitor", "chair"],
          "lounge": ["sofa", "tv", "lamp", "rug"]}
lines = []
for room, (x0, y0) in rooms.items():
    for index, label in enumerate(labels[room]):
        lines.append(json.dumps({"id": f"{room}/{index}", "label": label,
                                 "centroid": [x0 + index * 0.9, y0 + (index % 2) * 1.1, 0.4 + index * 0.3]}))
(out / "objects.jsonl").write_text("\n".join(lines) + "\n")
(out / "meta.json").write_text(json.dumps({"up_axis": "z", "frame": "ragmap_scene", "units": "metre"}))
