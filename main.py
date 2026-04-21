from __future__ import annotations

import base64
import copy
import json
import os
import struct
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

APP_TITLE = "MC3DS Model Editor"
APP_VERSION = "1.0"
TEXT_FORMAT_HINT = "Name\\nX, Y, Z\\nW, H, D\\n"


def _install_requirements() -> None:
    req = Path(__file__).with_name("requirements.txt")
    if not req.exists():
        raise FileNotFoundError("requirements.txt was not found next to main.py")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


try:
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:
    root = tk.Tk()
    root.withdraw()
    answer = messagebox.askyesno(
        APP_TITLE,
        "Some required Python packages are missing.\n\n"
        "Would you like this tool to install them from requirements.txt now?",
    )
    root.destroy()
    if answer:
        _install_requirements()
        os.execv(sys.executable, [sys.executable, __file__])
    raise

try:
    import stl
except Exception:
    stl = None

try:
    from pygltflib import Accessor, Buffer, BufferView, GLTF2, Mesh, Node, Primitive, Scene
except Exception:
    GLTF2 = None

try:
    from modules.bjson import BJSONFile
except Exception:
    BJSONFile = None


@dataclass
class Cuboid:
    geometry_key: str
    bone_name: str
    cube_index: int
    name: str
    origin: np.ndarray
    size: np.ndarray
    selected: bool = False
    uuid: str = ""

    def corners(self) -> list[list[float]]:
        x, y, z = self.origin.tolist()
        dx, dy, dz = self.size.tolist()
        return [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ]

    def clone(self) -> "Cuboid":
        return Cuboid(
            geometry_key=self.geometry_key,
            bone_name=self.bone_name,
            cube_index=self.cube_index,
            name=self.name,
            origin=self.origin.copy(),
            size=self.size.copy(),
            selected=False,
            uuid=self.uuid,
        )


@dataclass
class ModelDocument:
    source_type: str = "text"
    source_path: Optional[Path] = None
    data: dict[str, Any] | None = None
    cuboids: list[Cuboid] = field(default_factory=list)
    dirty: bool = False
    active_model_key: Optional[str] = None
    model_keys: list[str] = field(default_factory=list)

    @staticmethod
    def _numeric3(values: Any, fallback: tuple[float, float, float]) -> np.ndarray:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            return np.array(fallback, dtype=float)
        return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)

    @classmethod
    def from_text(cls, path: Path) -> "ModelDocument":
        with path.open("r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        cuboids: list[Cuboid] = []
        i = 0
        item_index = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            if i + 2 >= len(lines):
                raise ValueError("Text model format is incomplete near the end of the file.")
            name = lines[i].strip()
            origin = np.array([float(x.strip()) for x in lines[i + 1].split(",")], dtype=float)
            size = np.array([float(x.strip()) for x in lines[i + 2].split(",")], dtype=float)
            cuboids.append(
                Cuboid(
                    geometry_key="geometry.default",
                    bone_name="root",
                    cube_index=item_index,
                    name=name,
                    origin=origin,
                    size=size,
                    uuid=f"text:{item_index}",
                )
            )
            item_index += 1
            i += 3
            if i < len(lines) and not lines[i].strip():
                i += 1

        return cls(source_type="text", source_path=path, data=None, cuboids=cuboids, active_model_key="geometry.default", model_keys=["geometry.default"])

    @classmethod
    def from_json(cls, path: Path) -> "ModelDocument":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        cuboids, model_keys = cls._extract_cuboids_from_geometry_json(raw)
        return cls(source_type="json", source_path=path, data=raw, cuboids=cuboids, active_model_key=(model_keys[0] if model_keys else None), model_keys=model_keys)

    @classmethod
    def from_bjson(cls, path: Path) -> "ModelDocument":
        if BJSONFile is None:
            raise RuntimeError("BJSON support is unavailable because modules.bjson could not be imported.")
        raw_text = BJSONFile().open(str(path)).toJson(showDebug=False)
        raw = json.loads(raw_text)
        cuboids, model_keys = cls._extract_cuboids_from_geometry_json(raw)
        return cls(source_type="bjson", source_path=path, data=raw, cuboids=cuboids, active_model_key=(model_keys[0] if model_keys else None), model_keys=model_keys)

    @classmethod
    def from_bbmodel(cls, path: Path) -> "ModelDocument":
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        cuboids, model_key = cls._extract_cuboids_from_bbmodel(raw)
        return cls(source_type="bbmodel", source_path=path, data=raw, cuboids=cuboids, active_model_key=model_key, model_keys=[model_key])

    @staticmethod
    def _extract_cuboids_from_geometry_json(raw: dict[str, Any]) -> tuple[list[Cuboid], list[str]]:
        cuboids: list[Cuboid] = []
        model_keys: list[str] = []
        for geometry_key, geometry_data in raw.items():
            if not geometry_key.startswith("geometry.") or not isinstance(geometry_data, dict):
                continue
            model_keys.append(geometry_key)
            bones = geometry_data.get("bones", [])
            if not isinstance(bones, list):
                continue
            for bone in bones:
                if not isinstance(bone, dict):
                    continue
                bone_name = str(bone.get("name", "bone"))
                cubes = bone.get("cubes", []) or []
                if not isinstance(cubes, list):
                    continue
                for cube_index, cube in enumerate(cubes):
                    if not isinstance(cube, dict):
                        continue
                    origin = ModelDocument._numeric3(cube.get("origin"), (0.0, 0.0, 0.0))
                    size = ModelDocument._numeric3(cube.get("size"), (1.0, 1.0, 1.0))
                    display_name = f"{bone_name}[{cube_index}]"
                    cuboids.append(
                        Cuboid(
                            geometry_key=geometry_key,
                            bone_name=bone_name,
                            cube_index=cube_index,
                            name=display_name,
                            origin=origin,
                            size=size,
                            uuid=f"{geometry_key}|{bone_name}|{cube_index}",
                        )
                    )
        if not cuboids:
            raise ValueError("No editable cubes were found in the selected JSON/BJSON file.")
        return cuboids, model_keys

    @staticmethod
    def _extract_cuboids_from_bbmodel(raw: dict[str, Any]) -> tuple[list[Cuboid], str]:
        elements = raw.get("elements", [])
        if not isinstance(elements, list) or not elements:
            raise ValueError("No editable Blockbench elements were found in the selected .bbmodel file.")
        identifier = raw.get("model_identifier") or raw.get("name") or "blockbench_model"
        model_key = f"geometry.{identifier}"
        cuboids: list[Cuboid] = []
        for idx, element in enumerate(elements):
            if not isinstance(element, dict):
                continue
            origin = ModelDocument._numeric3(element.get("from"), (0.0, 0.0, 0.0))
            to_vec = ModelDocument._numeric3(element.get("to"), (origin[0] + 1.0, origin[1] + 1.0, origin[2] + 1.0))
            size = to_vec - origin
            bone_name = str(element.get("__group") or element.get("group") or "root")
            name = str(element.get("name") or f"element_{idx}")
            cuboids.append(
                Cuboid(
                    geometry_key=model_key,
                    bone_name=bone_name,
                    cube_index=idx,
                    name=name,
                    origin=origin,
                    size=size,
                    uuid=str(element.get("uuid") or f"bb:{idx}"),
                )
            )
        return cuboids, model_key

    def visible_cuboids(self) -> list[tuple[int, Cuboid]]:
        if self.active_model_key is None:
            return list(enumerate(self.cuboids))
        return [(idx, cube) for idx, cube in enumerate(self.cuboids) if cube.geometry_key == self.active_model_key]

    def set_active_model(self, model_key: Optional[str]) -> None:
        self.active_model_key = model_key

    def _apply_cuboids_to_json(self) -> dict[str, Any]:
        if self.data is None:
            raise RuntimeError("This document does not have JSON-backed data.")
        raw = copy.deepcopy(self.data)
        grouped: dict[tuple[str, str], list[Cuboid]] = {}
        for cube in self.cuboids:
            grouped.setdefault((cube.geometry_key, cube.bone_name), []).append(cube)

        for (geometry_key, bone_name), cubes in grouped.items():
            geometry = raw.get(geometry_key)
            if not isinstance(geometry, dict):
                continue
            bones = geometry.get("bones", [])
            if not isinstance(bones, list):
                continue
            existing_bone = None
            for bone in bones:
                if isinstance(bone, dict) and str(bone.get("name", "")) == bone_name:
                    existing_bone = bone
                    break
            if existing_bone is None:
                existing_bone = {"name": bone_name, "pivot": [0, 0, 0], "cubes": []}
                bones.append(existing_bone)
            existing_list = existing_bone.get("cubes", [])
            if not isinstance(existing_list, list):
                existing_list = []
            existing_bone["cubes"] = [
                {
                    **(existing_list[idx] if idx < len(existing_list) and isinstance(existing_list[idx], dict) else {}),
                    "origin": _round_trip_list(c.origin),
                    "size": _round_trip_list(c.size),
                }
                for idx, c in enumerate(sorted(cubes, key=lambda x: x.cube_index))
            ]
        return raw

    def _apply_cuboids_to_bbmodel(self) -> dict[str, Any]:
        if self.data is None:
            raise RuntimeError("This document does not have Blockbench-backed data.")
        raw = copy.deepcopy(self.data)
        raw.setdefault("meta", {"format_version": "4.0"})
        raw.setdefault("resolution", {"width": 64, "height": 64})
        if self.active_model_key and self.active_model_key.startswith("geometry."):
            raw["model_identifier"] = self.active_model_key.removeprefix("geometry.")
        if "name" not in raw:
            raw["name"] = raw.get("model_identifier", "Blockbench Model")
        original_elements = raw.get("elements", [])
        if not isinstance(original_elements, list):
            original_elements = []
        visible = [cube for _, cube in self.visible_cuboids()]
        elements: list[dict[str, Any]] = []
        for idx, cube in enumerate(visible):
            base = original_elements[idx] if idx < len(original_elements) and isinstance(original_elements[idx], dict) else {}
            element = copy.deepcopy(base)
            element["name"] = cube.name
            element["from"] = _round_trip_list(cube.origin)
            element["to"] = _round_trip_list(cube.origin + cube.size)
            element["uuid"] = cube.uuid or element.get("uuid") or f"bb:{idx}"
            if cube.bone_name != "root":
                element["__group"] = cube.bone_name
            elements.append(element)
        raw["elements"] = elements
        return raw

    def save_to(self, path: Path, target_type: Optional[str] = None) -> None:
        target_type = target_type or self.source_type
        if target_type == "text":
            self._save_text(path)
        elif target_type == "json":
            self._save_json(path)
        elif target_type == "bjson":
            self._save_bjson(path)
        elif target_type == "bbmodel":
            self._save_bbmodel(path)
        else:
            raise ValueError(f"Unsupported save type: {target_type}")
        self.source_path = path
        self.source_type = target_type
        if target_type in {"json", "bjson"}:
            self.data = self._apply_cuboids_to_json()
        elif target_type == "bbmodel":
            self.data = self._apply_cuboids_to_bbmodel()
        self.dirty = False

    def _save_text(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as f:
            for _, cube in self.visible_cuboids():
                f.write(f"{cube.name}\n")
                f.write(", ".join(_fmt_number(v) for v in cube.origin.tolist()) + "\n")
                f.write(", ".join(_fmt_number(v) for v in cube.size.tolist()) + "\n\n")

    def _save_json(self, path: Path) -> None:
        raw = self._apply_cuboids_to_json() if self.data is not None else _cuboids_to_basic_geometry_json([cube for _, cube in self.visible_cuboids()])
        with path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, indent=4)

    def _save_bjson(self, path: Path) -> None:
        if BJSONFile is None:
            raise RuntimeError("BJSON support is unavailable because modules.bjson could not be imported.")
        raw = self._apply_cuboids_to_json() if self.data is not None else _cuboids_to_basic_geometry_json([cube for _, cube in self.visible_cuboids()])
        json_text = json.dumps(raw, indent=4)
        bjson_file = BJSONFile()
        bjson_file.fromJson(json_text)
        with path.open("wb") as f:
            f.write(bjson_file.getData())

    def _save_bbmodel(self, path: Path) -> None:
        raw = self._apply_cuboids_to_bbmodel() if self.data is not None else _cuboids_to_basic_bbmodel([cube for _, cube in self.visible_cuboids()], self.active_model_key)
        with path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, indent=4)


class ModelEditorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(f"{APP_TITLE} v{APP_VERSION}")
        self.root.geometry("1460x860")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.document = ModelDocument()
        self.filtered_indices: list[int] = []
        self.current_index: Optional[int] = None
        self.drag_last_xy: Optional[tuple[int, int]] = None
        self.camera_azim = 35
        self.camera_elev = 25
        self.zoom_scale = 1.0
        self.status_var = tk.StringVar(value="Ready")
        self.search_var = tk.StringVar()
        self.model_info_var = tk.StringVar(value="No model loaded")

        self._build_style()
        self._build_ui()
        self._bind_shortcuts()
        self.new_document()

    def _build_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Treeview", rowheight=24)
        style.configure("Title.TLabel", font=("Segoe UI", 10, "bold"))

    def _build_ui(self) -> None:
        self._build_menu()

        toolbar = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        for text, command in [
            ("New", self.new_document),
            ("Open", self.open_model),
            ("Save", self.save_model),
            ("Save As", self.save_model_as),
            ("Add Cube", self.add_cube),
            ("Duplicate", self.duplicate_selected_cube),
            ("Delete", self.delete_selected_cube),
            ("Scale x2", lambda: self.scale_all(2.0)),
            ("Scale 0.5", lambda: self.scale_all(0.5)),
            ("Center", self.center_model),
        ]:
            ttk.Button(toolbar, text=text, command=command).pack(side=tk.LEFT, padx=3)

        ttk.Separator(self.root).pack(fill=tk.X, padx=8)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(main, padding=8)
        center = ttk.Frame(main, padding=8)
        right = ttk.Frame(main, padding=8)
        main.add(left, weight=28)
        main.add(center, weight=52)
        main.add(right, weight=26)

        ttk.Label(left, text="Model Explorer", style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, textvariable=self.model_info_var).pack(anchor="w", pady=(0, 4))

        model_row = ttk.Frame(left)
        model_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(model_row, text="Model").pack(side=tk.LEFT)
        self.active_model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_row, textvariable=self.active_model_var, state="readonly")
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        search_row = ttk.Frame(left)
        search_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Entry(search_row, textvariable=self.search_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(search_row, text="Filter", command=self.refresh_cube_tree).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(search_row, text="Clear", command=self.clear_filter).pack(side=tk.LEFT, padx=(4, 0))

        self.cube_tree = ttk.Treeview(left, columns=("bone", "origin", "size"), show="headings", selectmode="browse")
        self.cube_tree.heading("bone", text="Bone")
        self.cube_tree.heading("origin", text="Origin")
        self.cube_tree.heading("size", text="Size")
        self.cube_tree.column("bone", width=120, stretch=False)
        self.cube_tree.column("origin", width=120, stretch=False)
        self.cube_tree.column("size", width=110, stretch=False)
        self.cube_tree.pack(fill=tk.BOTH, expand=True)
        self.cube_tree.bind("<<TreeviewSelect>>", self.on_tree_select)
        self.cube_tree.bind("<Double-1>", lambda _e: self.focus_inspector())

        left_buttons = ttk.Frame(left)
        left_buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(left_buttons, text="Add", command=self.add_cube).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(left_buttons, text="Duplicate", command=self.duplicate_selected_cube).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(left_buttons, text="Delete", command=self.delete_selected_cube).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        ttk.Label(center, text="3D Preview", style="Title.TLabel").pack(anchor="w")
        preview_help = ttk.Label(center, text="Drag to rotate • Mouse wheel to zoom")
        preview_help.pack(anchor="w", pady=(0, 6))

        self.figure = plt.Figure(figsize=(7, 6), dpi=100)
        self.ax = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.figure, master=center)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self.on_scroll_plot)
        self.canvas.get_tk_widget().bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.get_tk_widget().bind("<B1-Motion>", self.on_drag_motion)
        self.canvas.get_tk_widget().bind("<ButtonRelease-1>", self.on_drag_end)

        preview_buttons = ttk.Frame(center)
        preview_buttons.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(preview_buttons, text="Reset View", command=self.reset_view).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(preview_buttons, text="Frame Model", command=self.redraw_preview).pack(side=tk.LEFT)

        ttk.Label(right, text="Inspector", style="Title.TLabel").pack(anchor="w")

        form = ttk.Frame(right)
        form.pack(fill=tk.X, pady=(6, 8))
        form.columnconfigure(1, weight=1)

        self.name_var = tk.StringVar()
        self.geometry_var = tk.StringVar()
        self.bone_var = tk.StringVar()
        self.origin_vars = [tk.StringVar(), tk.StringVar(), tk.StringVar()]
        self.size_vars = [tk.StringVar(), tk.StringVar(), tk.StringVar()]

        self._add_labeled_entry(form, 0, "Display Name", self.name_var)
        self._add_labeled_entry(form, 1, "Geometry", self.geometry_var, readonly=True)
        self._add_labeled_entry(form, 2, "Bone", self.bone_var, readonly=True)

        ttk.Label(form, text="Origin").grid(row=3, column=0, sticky="w", pady=4)
        origin_row = ttk.Frame(form)
        origin_row.grid(row=3, column=1, sticky="ew", pady=4)
        for idx, var in enumerate(self.origin_vars):
            ttk.Entry(origin_row, textvariable=var, width=8).pack(side=tk.LEFT, padx=(0, 4 if idx < 2 else 0))

        ttk.Label(form, text="Size").grid(row=4, column=0, sticky="w", pady=4)
        size_row = ttk.Frame(form)
        size_row.grid(row=4, column=1, sticky="ew", pady=4)
        for idx, var in enumerate(self.size_vars):
            ttk.Entry(size_row, textvariable=var, width=8).pack(side=tk.LEFT, padx=(0, 4 if idx < 2 else 0))

        inspector_buttons = ttk.Frame(right)
        inspector_buttons.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(inspector_buttons, text="Apply Changes", command=self.apply_inspector_changes).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 3))
        ttk.Button(inspector_buttons, text="Reset Fields", command=self.load_current_cube_into_form).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(3, 0))

        quick_actions = ttk.LabelFrame(right, text="Quick Actions", padding=8)
        quick_actions.pack(fill=tk.X)
        ttk.Button(quick_actions, text="Move +X", command=lambda: self.nudge_selected(1, 0, 0)).pack(fill=tk.X, pady=2)
        ttk.Button(quick_actions, text="Move -X", command=lambda: self.nudge_selected(-1, 0, 0)).pack(fill=tk.X, pady=2)
        ttk.Button(quick_actions, text="Move +Y", command=lambda: self.nudge_selected(0, 1, 0)).pack(fill=tk.X, pady=2)
        ttk.Button(quick_actions, text="Move -Y", command=lambda: self.nudge_selected(0, -1, 0)).pack(fill=tk.X, pady=2)
        ttk.Button(quick_actions, text="Move +Z", command=lambda: self.nudge_selected(0, 0, 1)).pack(fill=tk.X, pady=2)
        ttk.Button(quick_actions, text="Move -Z", command=lambda: self.nudge_selected(0, 0, -1)).pack(fill=tk.X, pady=2)

        export_box = ttk.LabelFrame(right, text="Export", padding=8)
        export_box.pack(fill=tk.X, pady=(10, 0))
        ttk.Button(export_box, text="OBJ", command=lambda: self.export_model("obj")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(export_box, text="STL", command=lambda: self.export_model("stl")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(export_box, text="PLY", command=lambda: self.export_model("ply")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(export_box, text="GLTF", command=lambda: self.export_model("gltf")).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        status = ttk.Frame(self.root, padding=(8, 4))
        status.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Separator(status, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(0, 4))
        ttk.Label(status, textvariable=self.status_var).pack(anchor="w")

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        self.root.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="New", command=self.new_document, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.open_model, accelerator="Ctrl+O")
        file_menu.add_command(label="Import Models Into Current File...", command=self.import_models_into_current)
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.save_model, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.save_model_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export as JSON...", command=lambda: self.save_as_specific("json"))
        file_menu.add_command(label="Export as BJSON...", command=lambda: self.save_as_specific("bjson"))
        file_menu.add_command(label="Export as Text...", command=lambda: self.save_as_specific("text"))
        file_menu.add_command(label="Export as Blockbench...", command=lambda: self.save_as_specific("bbmodel"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_close)
        menu.add_cascade(label="File", menu=file_menu)

        edit_menu = tk.Menu(menu, tearoff=False)
        edit_menu.add_command(label="Add Cube", command=self.add_cube)
        edit_menu.add_command(label="Duplicate Selected Cube", command=self.duplicate_selected_cube)
        edit_menu.add_command(label="Delete Selected Cube", command=self.delete_selected_cube)
        edit_menu.add_separator()
        edit_menu.add_command(label="Scale All x2", command=lambda: self.scale_all(2.0))
        edit_menu.add_command(label="Scale All 0.5", command=lambda: self.scale_all(0.5))
        edit_menu.add_command(label="Center Model", command=self.center_model)
        menu.add_cascade(label="Edit", menu=edit_menu)

        help_menu = tk.Menu(menu, tearoff=False)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="Text Format Help", command=self.show_text_format_help)
        menu.add_cascade(label="Help", menu=help_menu)

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-n>", lambda _e: self.new_document())
        self.root.bind("<Control-o>", lambda _e: self.open_model())
        self.root.bind("<Control-s>", lambda _e: self.save_model())
        self.root.bind("<Control-S>", lambda _e: self.save_model_as())
        self.root.bind("<Delete>", lambda _e: self.delete_selected_cube())

    def _add_labeled_entry(self, parent: ttk.Frame, row: int, label: str, var: tk.StringVar, readonly: bool = False) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var)
        if readonly:
            entry.state(["readonly"])
        entry.grid(row=row, column=1, sticky="ew", pady=4)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)

    def clear_filter(self) -> None:
        self.search_var.set("")
        self.refresh_cube_tree()

    def focus_inspector(self) -> None:
        self.root.focus_force()

    def new_document(self) -> None:
        if not self.confirm_discard_changes():
            return
        default_cube = Cuboid(
            geometry_key="geometry.default",
            bone_name="root",
            cube_index=0,
            name="cube0",
            origin=np.array([0.0, 0.0, 0.0], dtype=float),
            size=np.array([8.0, 8.0, 8.0], dtype=float),
            uuid="new:0",
        )
        self.document = ModelDocument(source_type="text", source_path=None, data=None, cuboids=[default_cube], dirty=False, active_model_key="geometry.default", model_keys=["geometry.default"])
        self.current_index = 0
        self.refresh_cube_tree(select_index=0)
        self.redraw_preview()
        self.load_current_cube_into_form()
        self.set_status("Created a new model.")

    def open_model(self) -> None:
        if not self.confirm_discard_changes():
            return
        file_path = filedialog.askopenfilename(
            title="Open model",
            filetypes=[
                ("Supported models", "*.txt *.json *.bjson *.bbmodel"),
                ("Text models", "*.txt"),
                ("JSON models", "*.json"),
                ("BJSON models", "*.bjson"),
                ("Blockbench models", "*.bbmodel"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        path = Path(file_path)
        try:
            self.document = self._load_document_from_path(path)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open the selected file.\n\n{exc}")
            return
        self.current_index = 0 if self.document.cuboids else None
        self.refresh_cube_tree(select_index=self.current_index)
        self.redraw_preview()
        self.load_current_cube_into_form()
        self.set_status(f"Opened: {path.name}")

    def _load_document_from_path(self, path: Path) -> ModelDocument:
        suffix = path.suffix.lower()
        if suffix == ".txt":
            return ModelDocument.from_text(path)
        if suffix == ".json":
            return ModelDocument.from_json(path)
        if suffix == ".bjson":
            return ModelDocument.from_bjson(path)
        if suffix == ".bbmodel":
            return ModelDocument.from_bbmodel(path)
        raise ValueError("Unsupported file type.")

    def _ensure_geometry_document_for_merge(self) -> None:
        if self.document.source_type in {"json", "bjson"} and isinstance(self.document.data, dict):
            return

        raw = _cuboids_to_basic_geometry_json(list(self.document.cuboids))
        self.document.data = raw
        self.document.source_type = "json" if self.document.source_type == "text" else "json"
        self.document.model_keys = [key for key in raw.keys() if isinstance(key, str) and key.startswith("geometry.")]
        self.document.active_model_key = self.document.active_model_key or (self.document.model_keys[0] if self.document.model_keys else None)

    def import_models_into_current(self) -> None:
        if not self.document.cuboids and self.document.data is None:
            messagebox.showwarning(APP_TITLE, "Open or create a target model first.")
            return

        file_path = filedialog.askopenfilename(
            title="Import models into current file",
            filetypes=[
                ("Supported models", "*.txt *.json *.bjson *.bbmodel"),
                ("Text models", "*.txt"),
                ("JSON models", "*.json"),
                ("BJSON models", "*.bjson"),
                ("Blockbench models", "*.bbmodel"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            imported = self._load_document_from_path(Path(file_path))
            self._ensure_geometry_document_for_merge()
            imported_raw = _document_to_geometry_json(imported)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not import models from the selected file.\n\n{exc}")
            return

        existing_names = set(self.document.model_keys)
        incoming_names = [key for key in imported_raw.keys() if isinstance(key, str) and key.startswith("geometry.")]
        collisions = sorted(name for name in incoming_names if name in existing_names)
        replace_existing = False
        if collisions:
            replace_existing = messagebox.askyesno(
                APP_TITLE,
                "One or more imported model names already exist in the current file.\n\n"
                + "Choose Yes to overwrite same-named models.\n"
                + "Choose No to keep both by importing duplicates with a new name.\n\n"
                + "Conflicting models:\n"
                + "\n".join(collisions[:12]),
            )

        rename_map: dict[str, str] = {}
        for model_name in incoming_names:
            final_name = model_name
            if model_name in existing_names and not replace_existing:
                final_name = _make_unique_model_name(model_name, existing_names)
            rename_map[model_name] = final_name
            existing_names.add(final_name)

        target_raw = self.document._apply_cuboids_to_json() if self.document.data is not None else {}

        for old_name, new_name in rename_map.items():
            geometry_data = copy.deepcopy(imported_raw.get(old_name, {}))
            if replace_existing or new_name not in target_raw:
                target_raw[new_name] = geometry_data
            else:
                target_raw[new_name] = geometry_data

        self.document.data = target_raw
        cuboids, model_keys = ModelDocument._extract_cuboids_from_geometry_json(target_raw)
        for cube in cuboids:
            cube.geometry_key = rename_map.get(cube.geometry_key, cube.geometry_key)
            cube.uuid = f"{cube.geometry_key}|{cube.bone_name}|{cube.cube_index}"
        self.document.cuboids = cuboids
        self.document.model_keys = [rename_map.get(key, key) for key in model_keys]
        self.document.model_keys = list(dict.fromkeys(self.document.model_keys))
        preferred = self.document.active_model_key
        if preferred in self.document.model_keys:
            self.document.active_model_key = preferred
        elif rename_map:
            self.document.active_model_key = next(iter(rename_map.values()))
        elif self.document.model_keys:
            self.document.active_model_key = self.document.model_keys[0]
        self.current_index = self.document.visible_cuboids()[0][0] if self.document.visible_cuboids() else None
        self.mark_dirty()
        self.refresh_cube_tree(select_index=self.current_index)
        self.load_current_cube_into_form()
        self.redraw_preview()
        imported_count = len(rename_map)
        self.set_status(f"Imported {imported_count} model(s) from {Path(file_path).name}.")

    def save_model(self) -> None:
        if self.document.source_path is None:
            self.save_model_as()
            return
        try:
            self.document.save_to(self.document.source_path, self.document.source_type)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not save the model.\n\n{exc}")
            return
        self.refresh_title()
        self.set_status(f"Saved: {self.document.source_path.name}")

    def save_model_as(self) -> None:
        self._save_via_dialog(default_type=self.document.source_type or "text")

    def save_as_specific(self, target_type: str) -> None:
        self._save_via_dialog(default_type=target_type)

    def _save_via_dialog(self, default_type: str) -> None:
        filetypes_map = {
            "text": (("Text models", "*.txt"),),
            "json": (("JSON models", "*.json"),),
            "bjson": (("BJSON models", "*.bjson"),),
            "bbmodel": (("Blockbench models", "*.bbmodel"),),
        }
        extension_map = {"text": ".txt", "json": ".json", "bjson": ".bjson", "bbmodel": ".bbmodel"}
        file_path = filedialog.asksaveasfilename(
            title="Save model as",
            defaultextension=extension_map[default_type],
            filetypes=[*filetypes_map[default_type], ("All files", "*.*")],
        )
        if not file_path:
            return
        try:
            self.document.save_to(Path(file_path), default_type)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not save the model.\n\n{exc}")
            return
        self.refresh_title()
        self.refresh_cube_tree(select_index=self.current_index)
        self.set_status(f"Saved as {Path(file_path).name}")

    def refresh_title(self) -> None:
        suffix = " *" if self.document.dirty else ""
        name = self.document.source_path.name if self.document.source_path else "Untitled"
        self.root.title(f"{APP_TITLE} v{APP_VERSION} — {name}{suffix}")

    def refresh_cube_tree(self, select_index: Optional[int] = None) -> None:
        for item in self.cube_tree.get_children():
            self.cube_tree.delete(item)

        query = self.search_var.get().strip().lower()
        self.filtered_indices = []

        for idx, cube in self.document.visible_cuboids():
            hay = f"{cube.name} {cube.bone_name} {cube.geometry_key}".lower()
            if query and query not in hay:
                continue
            self.filtered_indices.append(idx)
            self.cube_tree.insert(
                "",
                tk.END,
                iid=str(idx),
                values=(
                    cube.name,
                    _triple_to_text(cube.origin),
                    _triple_to_text(cube.size),
                ),
            )

        if select_index is not None and str(select_index) in self.cube_tree.get_children():
            self.cube_tree.selection_set(str(select_index))
            self.cube_tree.focus(str(select_index))
            self.cube_tree.see(str(select_index))
            self.current_index = select_index
        elif self.current_index is not None and str(self.current_index) in self.cube_tree.get_children():
            self.cube_tree.selection_set(str(self.current_index))
        elif self.filtered_indices:
            self.current_index = self.filtered_indices[0]
            self.cube_tree.selection_set(str(self.current_index))
        else:
            self.current_index = None

        loaded_name = self.document.source_path.name if self.document.source_path else "Untitled"
        visible_count = len(self.document.visible_cuboids())
        total_count = len(self.document.cuboids)
        self.model_info_var.set(f"{loaded_name} • showing {visible_count} / {total_count} cube(s)")
        self.model_combo["values"] = self.document.model_keys or [""]
        self.active_model_var.set(self.document.active_model_key or "")
        self.refresh_title()

    def on_model_change(self, _event: tk.Event | None = None) -> None:
        selected_model = self.active_model_var.get().strip() or None
        self.document.set_active_model(selected_model)
        visible = self.document.visible_cuboids()
        self.current_index = visible[0][0] if visible else None
        self.refresh_cube_tree(select_index=self.current_index)
        self.load_current_cube_into_form()
        self.redraw_preview()
        if selected_model:
            self.set_status(f"Switched to model: {selected_model}")

    def on_tree_select(self, _event: tk.Event | None = None) -> None:
        selected = self.cube_tree.selection()
        if not selected:
            return
        self.current_index = int(selected[0])
        self.load_current_cube_into_form()
        self.redraw_preview()

    def current_cube(self) -> Optional[Cuboid]:
        if self.current_index is None:
            return None
        if not (0 <= self.current_index < len(self.document.cuboids)):
            return None
        return self.document.cuboids[self.current_index]

    def load_current_cube_into_form(self) -> None:
        cube = self.current_cube()
        if cube is None:
            self.name_var.set("")
            self.geometry_var.set("")
            self.bone_var.set("")
            for var in self.origin_vars + self.size_vars:
                var.set("")
            return
        self.name_var.set(cube.name)
        self.geometry_var.set(cube.geometry_key)
        self.bone_var.set(cube.bone_name)
        for idx in range(3):
            self.origin_vars[idx].set(_fmt_number(float(cube.origin[idx])))
            self.size_vars[idx].set(_fmt_number(float(cube.size[idx])))

    def apply_inspector_changes(self) -> None:
        cube = self.current_cube()
        if cube is None:
            messagebox.showwarning(APP_TITLE, "Select a cube first.")
            return
        try:
            new_origin = np.array([float(v.get().strip()) for v in self.origin_vars], dtype=float)
            new_size = np.array([float(v.get().strip()) for v in self.size_vars], dtype=float)
        except ValueError:
            messagebox.showerror(APP_TITLE, "Origin and Size must use valid numbers.")
            return
        if np.any(new_size == 0):
            messagebox.showerror(APP_TITLE, "Cube size values cannot be zero.")
            return

        cube.name = self.name_var.get().strip() or cube.name
        cube.origin = new_origin
        cube.size = new_size
        self.mark_dirty()
        self.refresh_cube_tree(select_index=self.current_index)
        self.redraw_preview()
        self.set_status(f"Updated {cube.name}")

    def mark_dirty(self) -> None:
        self.document.dirty = True
        self.refresh_title()

    def add_cube(self) -> None:
        if self.document.source_type in {"json", "bjson", "bbmodel"} and self.document.data is not None:
            geometry_key = self.document.active_model_key or (self.document.model_keys[0] if self.document.model_keys else "geometry.default")
            bone_name = self._choose_bone_for_active_model(geometry_key)
            if bone_name is None:
                return
        else:
            geometry_key, bone_name = "geometry.default", "root"

        next_index = sum(1 for c in self.document.cuboids if c.geometry_key == geometry_key and c.bone_name == bone_name)
        new_cube = Cuboid(
            geometry_key=geometry_key,
            bone_name=bone_name,
            cube_index=next_index,
            name=f"{bone_name}[{next_index}]",
            origin=np.array([0.0, 0.0, 0.0], dtype=float),
            size=np.array([4.0, 4.0, 4.0], dtype=float),
            uuid=f"new:{time.time()}:{next_index}",
        )
        self.document.cuboids.append(new_cube)
        self._reindex_group(geometry_key, bone_name)
        self.current_index = len(self.document.cuboids) - 1
        self.mark_dirty()
        self.refresh_cube_tree(select_index=self.current_index)
        self.load_current_cube_into_form()
        self.redraw_preview()
        self.set_status("Added a new cube.")

    def duplicate_selected_cube(self) -> None:
        cube = self.current_cube()
        if cube is None:
            messagebox.showwarning(APP_TITLE, "Select a cube first.")
            return
        cloned = cube.clone()
        cloned.origin = cube.origin + np.array([1.0, 1.0, 1.0], dtype=float)
        cloned.name = f"{cube.bone_name}[copy]"
        cloned.uuid = f"dup:{time.time()}"
        self.document.cuboids.append(cloned)
        self._reindex_group(cloned.geometry_key, cloned.bone_name)
        self.current_index = self.document.cuboids.index(cloned)
        self.mark_dirty()
        self.refresh_cube_tree(select_index=self.current_index)
        self.load_current_cube_into_form()
        self.redraw_preview()
        self.set_status(f"Duplicated {cube.name}")

    def delete_selected_cube(self) -> None:
        cube = self.current_cube()
        if cube is None:
            return
        if len(self.document.cuboids) == 1:
            messagebox.showwarning(APP_TITLE, "You need at least one cube in the model.")
            return
        if not messagebox.askyesno(APP_TITLE, f"Delete {cube.name}?"):
            return
        geometry_key, bone_name = cube.geometry_key, cube.bone_name
        del self.document.cuboids[self.current_index]
        self._reindex_group(geometry_key, bone_name)
        self.current_index = min(self.current_index, len(self.document.cuboids) - 1)
        self.mark_dirty()
        self.refresh_cube_tree(select_index=self.current_index)
        self.load_current_cube_into_form()
        self.redraw_preview()
        self.set_status("Cube deleted.")

    def _reindex_group(self, geometry_key: str, bone_name: str) -> None:
        group = [c for c in self.document.cuboids if c.geometry_key == geometry_key and c.bone_name == bone_name]
        for idx, cube in enumerate(group):
            cube.cube_index = idx
            if "[copy]" not in cube.name:
                cube.name = f"{bone_name}[{idx}]"

    def nudge_selected(self, dx: float, dy: float, dz: float) -> None:
        cube = self.current_cube()
        if cube is None:
            return
        cube.origin = cube.origin + np.array([dx, dy, dz], dtype=float)
        self.mark_dirty()
        self.load_current_cube_into_form()
        self.refresh_cube_tree(select_index=self.current_index)
        self.redraw_preview()

    def scale_all(self, factor: float) -> None:
        visible = self.document.visible_cuboids()
        if not visible:
            return
        for _, cube in visible:
            cube.origin = cube.origin * factor
            cube.size = cube.size * factor
        self.mark_dirty()
        self.load_current_cube_into_form()
        self.refresh_cube_tree(select_index=self.current_index)
        self.redraw_preview()
        self.set_status(f"Scaled entire model by {factor}.")

    def center_model(self) -> None:
        if not self.document.cuboids:
            return
        mins, maxs = self._get_model_bounds()
        center = (mins + maxs) / 2.0
        for _, cube in self.document.visible_cuboids():
            cube.origin = cube.origin - center
        self.mark_dirty()
        self.load_current_cube_into_form()
        self.refresh_cube_tree(select_index=self.current_index)
        self.redraw_preview()
        self.set_status("Centered the model around the origin.")

    def _get_model_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        visible = [cube for _, cube in self.document.visible_cuboids()]
        if not visible:
            return np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
        mins = np.min(np.array([c.origin for c in visible]), axis=0)
        maxs = np.max(np.array([c.origin + c.size for c in visible]), axis=0)
        return mins, maxs

    def redraw_preview(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#1f1f1f")
        self.figure.patch.set_facecolor("#1f1f1f")
        self.ax.view_init(self.camera_elev, self.camera_azim)

        selected = self.current_cube()
        for idx, cube in self.document.visible_cuboids():
            corners = cube.corners()
            faces = [
                [corners[0], corners[1], corners[2], corners[3]],
                [corners[4], corners[5], corners[6], corners[7]],
                [corners[0], corners[1], corners[5], corners[4]],
                [corners[2], corners[3], corners[7], corners[6]],
                [corners[0], corners[3], corners[7], corners[4]],
                [corners[1], corners[2], corners[6], corners[5]],
            ]
            is_selected = selected is not None and idx == self.current_index
            poly = Poly3DCollection(
                faces,
                facecolors=("#49b6ff" if is_selected else "#62d0c7"),
                edgecolors="#ff9090",
                linewidths=0.8,
                alpha=(0.45 if is_selected else 0.18),
            )
            self.ax.add_collection3d(poly)
            label_pos = cube.origin + (cube.size / 2.0)
            self.ax.text(label_pos[0], label_pos[1], label_pos[2], cube.name, color=("white" if is_selected else "#d9d9d9"), fontsize=8)

        mins, maxs = self._get_model_bounds()
        max_range = float(np.max(maxs - mins))
        if max_range <= 0:
            max_range = 1.0
        max_range *= self.zoom_scale
        mid = (mins + maxs) / 2.0
        half = max_range / 2.0
        self.ax.set_xlim(mid[0] - half, mid[0] + half)
        self.ax.set_ylim(mid[1] - half, mid[1] + half)
        self.ax.set_zlim(mid[2] - half, mid[2] + half)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.grid(True, alpha=0.2)
        self.canvas.draw_idle()

    def reset_view(self) -> None:
        self.camera_azim = 35
        self.camera_elev = 25
        self.zoom_scale = 1.0
        self.redraw_preview()
        self.set_status("View reset.")

    def on_scroll_plot(self, event: Any) -> None:
        if getattr(event, "button", None) == "up":
            self.zoom_scale = max(0.1, self.zoom_scale * 0.9)
        else:
            self.zoom_scale = min(10.0, self.zoom_scale * 1.1)
        self.redraw_preview()

    def on_drag_start(self, event: tk.Event) -> None:
        self.drag_last_xy = (event.x, event.y)

    def on_drag_motion(self, event: tk.Event) -> None:
        if self.drag_last_xy is None:
            return
        last_x, last_y = self.drag_last_xy
        self.camera_azim += (event.x - last_x) * 0.5
        self.camera_elev -= (event.y - last_y) * 0.5
        self.drag_last_xy = (event.x, event.y)
        self.redraw_preview()

    def on_drag_end(self, _event: tk.Event) -> None:
        self.drag_last_xy = None

    def export_model(self, kind: str) -> None:
        if not self.document.visible_cuboids():
            messagebox.showwarning(APP_TITLE, "There is no visible model to export.")
            return
        try:
            if kind == "obj":
                self.export_obj()
            elif kind == "stl":
                self.export_stl()
            elif kind == "ply":
                self.export_ply()
            elif kind == "gltf":
                self.export_gltf()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Export failed.\n\n{exc}")

    def export_obj(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".obj", filetypes=[("OBJ files", "*.obj")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Exported by MC3DS Model Editor Plus\n")
            vertex_offset = 1
            for _, cube in self.document.visible_cuboids():
                vertices = cube.corners()
                f.write(f"o {cube.name}\n")
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                faces = [
                    (0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                    (2, 3, 7, 6), (0, 3, 7, 4), (1, 2, 6, 5),
                ]
                for face in faces:
                    a, b, c, d = [vertex_offset + idx for idx in face]
                    f.write(f"f {a} {b} {c} {d}\n")
                vertex_offset += 8
        self.set_status(f"Exported OBJ: {Path(path).name}")

    def export_stl(self) -> None:
        if stl is None:
            raise RuntimeError("numpy-stl is not installed, so STL export is unavailable.")
        path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])
        if not path:
            return
        triangles: list[np.ndarray] = []
        for _, cube in self.document.visible_cuboids():
            v = np.array(cube.corners(), dtype=float)
            triangles.extend([
                np.array([v[0], v[1], v[2]]), np.array([v[0], v[2], v[3]]),
                np.array([v[4], v[5], v[6]]), np.array([v[4], v[6], v[7]]),
                np.array([v[0], v[1], v[5]]), np.array([v[0], v[5], v[4]]),
                np.array([v[2], v[3], v[7]]), np.array([v[2], v[7], v[6]]),
                np.array([v[0], v[3], v[7]]), np.array([v[0], v[7], v[4]]),
                np.array([v[1], v[2], v[6]]), np.array([v[1], v[6], v[5]]),
            ])
        mesh = stl.mesh.Mesh(np.zeros(len(triangles), dtype=stl.mesh.Mesh.dtype))
        for i, tri in enumerate(triangles):
            mesh.vectors[i] = tri
        mesh.save(path)
        self.set_status(f"Exported STL: {Path(path).name}")

    def export_ply(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".ply", filetypes=[("PLY files", "*.ply")])
        if not path:
            return
        vertices: list[tuple[float, float, float]] = []
        faces: list[tuple[int, int, int, int]] = []
        for _, cube in self.document.visible_cuboids():
            start = len(vertices)
            verts = [tuple(v) for v in cube.corners()]
            vertices.extend(verts)
            faces.extend([
                (start + 0, start + 1, start + 2, start + 3),
                (start + 4, start + 5, start + 6, start + 7),
                (start + 0, start + 1, start + 5, start + 4),
                (start + 2, start + 3, start + 7, start + 6),
                (start + 0, start + 3, start + 7, start + 4),
                (start + 1, start + 2, start + 6, start + 5),
            ])
        with open(path, "w", encoding="utf-8") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\nend_header\n")
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"4 {face[0]} {face[1]} {face[2]} {face[3]}\n")
        self.set_status(f"Exported PLY: {Path(path).name}")

    def export_gltf(self) -> None:
        if GLTF2 is None:
            raise RuntimeError("pygltflib is not installed, so GLTF export is unavailable.")
        path = filedialog.asksaveasfilename(defaultextension=".gltf", filetypes=[("GLTF files", "*.gltf")])
        if not path:
            return
        vertices: list[float] = []
        indices: list[int] = []
        vertex_offset = 0
        face_indices = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (2, 3, 7), (2, 7, 6),
            (0, 3, 7), (0, 7, 4),
            (1, 2, 6), (1, 6, 5),
        ]
        for _, cube in self.document.visible_cuboids():
            cube_vertices = cube.corners()
            for v in cube_vertices:
                vertices.extend(v)
            for tri in face_indices:
                indices.extend([vertex_offset + tri[0], vertex_offset + tri[1], vertex_offset + tri[2]])
            vertex_offset += 8

        vertices_bytes = struct.pack(f"{len(vertices)}f", *vertices)
        indices_bytes = struct.pack(f"{len(indices)}I", *indices)
        blob = vertices_bytes + indices_bytes

        gltf = GLTF2()
        gltf.scenes = [Scene(nodes=[0])]
        gltf.scene = 0
        gltf.nodes = [Node(mesh=0)]
        gltf.meshes = [Mesh(primitives=[Primitive(attributes={"POSITION": 0}, indices=1)])]
        gltf.buffers = [Buffer(byteLength=len(blob), uri="data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii"))]
        gltf.bufferViews = [
            BufferView(buffer=0, byteOffset=0, byteLength=len(vertices_bytes), target=34962),
            BufferView(buffer=0, byteOffset=len(vertices_bytes), byteLength=len(indices_bytes), target=34963),
        ]
        gltf.accessors = [
            Accessor(bufferView=0, byteOffset=0, componentType=5126, count=len(vertices) // 3, type="VEC3"),
            Accessor(bufferView=1, byteOffset=0, componentType=5125, count=len(indices), type="SCALAR"),
        ]
        gltf.save(path)
        self.set_status(f"Exported GLTF: {Path(path).name}")

    def _choose_bone_for_active_model(self, geometry_key: str) -> Optional[str]:
        if self.document.source_type == "bbmodel":
            existing = sorted({cube.bone_name for _, cube in self.document.visible_cuboids()})
            options = existing or ["root"]
            prompt = "Enter the target group/bone name for the new Blockbench cube:\n\n" + "\n".join(options[:20])
            selection = simpledialog.askstring(APP_TITLE, prompt, initialvalue=(options[0] if options else "root"))
            return selection.strip() if selection else None

        options: list[str] = []
        if self.document.data:
            geometry_data = self.document.data.get(geometry_key, {})
            if isinstance(geometry_data, dict):
                bones = geometry_data.get("bones", [])
                if isinstance(bones, list):
                    for bone in bones:
                        if isinstance(bone, dict):
                            options.append(str(bone.get("name", "bone")))
        options = sorted(set(options))
        if not options:
            return "root"
        prompt = "Enter the exact bone name for the active model:\n\n" + "\n".join(options[:20])
        selection = simpledialog.askstring(APP_TITLE, prompt, initialvalue=options[0])
        if not selection:
            return None
        if selection.strip() in options:
            return selection.strip()
        messagebox.showerror(APP_TITLE, "That bone name did not match one of the available bones.")
        return None


    def show_about(self) -> None:
        messagebox.showinfo(
            APP_TITLE,
            f"{APP_TITLE} v{APP_VERSION} - Cracko298\n\n"
            "• Better UI/UX.\n"
            "• Safer Save System.\n"
            "• Search & Duplication.\n"
            "• Selectable Models & Filters.\n"
            "• Blockbench (.bbmodel) Support.\n"
            "• Model importing/merging into JSON/BJSON.\n"
            "• Easier exporting into Standard Model Formats.",
        )

    def show_text_format_help(self) -> None:
        messagebox.showinfo(APP_TITLE, f"Text model format:\n\n{TEXT_FORMAT_HINT}")

    def confirm_discard_changes(self) -> bool:
        if not self.document.dirty:
            return True
        return messagebox.askyesno(APP_TITLE, "You have unsaved changes. Continue and discard them?")

    def on_close(self) -> None:
        if not self.confirm_discard_changes():
            return
        self.root.destroy()


def _document_to_geometry_json(document: ModelDocument) -> dict[str, Any]:
    if document.source_type in {"json", "bjson"} and isinstance(document.data, dict):
        return document._apply_cuboids_to_json()

    grouped: dict[str, list[Cuboid]] = {}
    for cube in document.cuboids:
        grouped.setdefault(cube.geometry_key or "geometry.default", []).append(cube)

    normalized: list[Cuboid] = []
    for geometry_key, cubes in grouped.items():
        if not str(geometry_key).startswith("geometry."):
            geometry_key = f"geometry.{geometry_key}"
        bone_counts: dict[str, int] = {}
        for cube in cubes:
            bone_name = cube.bone_name or "root"
            cube_index = bone_counts.get(bone_name, 0)
            bone_counts[bone_name] = cube_index + 1
            normalized.append(
                Cuboid(
                    geometry_key=geometry_key,
                    bone_name=bone_name,
                    cube_index=cube_index,
                    name=cube.name,
                    origin=cube.origin.copy(),
                    size=cube.size.copy(),
                    uuid=cube.uuid,
                )
            )
    return _cuboids_to_basic_geometry_json(normalized)


def _make_unique_model_name(base_name: str, existing_names: set[str]) -> str:
    if base_name not in existing_names:
        return base_name
    stem = base_name
    counter = 2
    while True:
        candidate = f"{stem}_{counter}"
        if candidate not in existing_names:
            return candidate
        counter += 1


def _round_trip_list(values: np.ndarray) -> list[int | float]:
    out: list[int | float] = []
    for value in values.tolist():
        value = float(value)
        out.append(int(value) if value.is_integer() else value)
    return out


def _cuboids_to_basic_geometry_json(cuboids: list[Cuboid]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Cuboid]] = {}
    for cube in cuboids:
        grouped.setdefault((cube.geometry_key, cube.bone_name), []).append(cube)

    geometry_map: dict[str, dict[str, Any]] = {}
    for (geometry_key, bone_name), cubes in grouped.items():
        geometry = geometry_map.setdefault(geometry_key, {"bones": []})
        geometry["bones"].append(
            {
                "name": bone_name,
                "pivot": [0, 0, 0],
                "cubes": [{"origin": _round_trip_list(c.origin), "size": _round_trip_list(c.size)} for c in cubes],
            }
        )
    return geometry_map or {"geometry.default": {"bones": []}}


def _cuboids_to_basic_bbmodel(cuboids: list[Cuboid], active_model_key: Optional[str]) -> dict[str, Any]:
    identifier = (active_model_key or "geometry.blockbench_model").removeprefix("geometry.")
    return {
        "meta": {"format_version": "4.0"},
        "name": identifier,
        "model_identifier": identifier,
        "resolution": {"width": 64, "height": 64},
        "elements": [
            {
                "name": cube.name,
                "from": _round_trip_list(cube.origin),
                "to": _round_trip_list(cube.origin + cube.size),
                "uuid": cube.uuid or f"bb:{idx}",
                **({"__group": cube.bone_name} if cube.bone_name != "root" else {}),
            }
            for idx, cube in enumerate(cuboids)
        ],
    }


def _fmt_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


def _triple_to_text(values: np.ndarray) -> str:
    return ", ".join(_fmt_number(float(v)) for v in values.tolist())


def main() -> None:
    root = tk.Tk()
    app = ModelEditorApp(root)
    app.refresh_title()
    root.mainloop()


if __name__ == "__main__":
    main()
