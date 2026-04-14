"""Thread-safe schema DAG for dynamic schema discovery during extraction."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Optional

from pydantic import BaseModel, Field as PydanticField

from sliders.llm_models import Field, Table, Tables


class ForeignKeyRef(BaseModel):
    """A foreign-key reference from fields in this table to a primary key in another table."""

    fields: list[str] = PydanticField(..., description="Column names in the child table that form the FK.")
    references_table: str = PydanticField(..., description="Name of the parent table whose PK is referenced.")
    references_fields: list[str] = PydanticField(
        ..., description="Column names in the parent table that form the referenced PK."
    )


class SchemaNode(BaseModel):
    """A single table in the schema DAG, with PK and FK metadata."""

    table: Table
    primary_key: Optional[list[str]] = PydanticField(
        default=None,
        description="Column names forming the primary key. None until selected post-extraction.",
    )
    foreign_keys: list[ForeignKeyRef] = PydanticField(
        default_factory=list,
        description="Foreign-key edges pointing to parent tables in the DAG.",
    )


_DOCUMENT_METADATA_NODE = SchemaNode(
    table=Table(
        name="DocumentMetadata",
        description="Root metadata for every chunk extracted from a document.",
        fields=[
            Field(
                name="document_name",
                data_type="str",
                enum_values=None,
                unit=None,
                scale=None,
                description="Name / title of the source document.",
                required=True,
                normalization=None,
            ),
            Field(
                name="chunk_id",
                data_type="int",
                enum_values=None,
                unit=None,
                scale=None,
                description="Zero-based index of the chunk within the document.",
                required=True,
                normalization=None,
            ),
            Field(
                name="page_number",
                data_type="int",
                enum_values=None,
                unit=None,
                scale=None,
                description="Page number that the chunk corresponds to (may equal chunk_id).",
                required=False,
                normalization=None,
            ),
            Field(
                name="text_header",
                data_type="str",
                enum_values=None,
                unit=None,
                scale=None,
                description="Section header(s) associated with this chunk, if any.",
                required=False,
                normalization=None,
            ),
        ],
    ),
    primary_key=["document_name", "chunk_id"],
    foreign_keys=[],
)


class SchemaDAG:
    """Concurrent-safe directed acyclic graph of normalised schemas.

    Every node is a :class:`SchemaNode` keyed by table name.  Edges are the
    :class:`ForeignKeyRef` entries inside each node, always pointing *from*
    child to parent.

    Thread-safety is provided via an :class:`asyncio.Lock`.  Writers must call
    :meth:`add_schema` (acquires the lock internally).  Readers should call
    :meth:`get_snapshot` for a consistent copy or :meth:`get_prompt_repr` for
    a string representation suitable for LLM prompts.
    """

    def __init__(self, seed_root: bool = True) -> None:
        self._nodes: dict[str, SchemaNode] = {}
        self._lock = asyncio.Lock()
        if seed_root:
            self._nodes[_DOCUMENT_METADATA_NODE.table.name] = _DOCUMENT_METADATA_NODE.model_copy(deep=True)

    # ------------------------------------------------------------------
    # Write operations (lock-protected)
    # ------------------------------------------------------------------

    async def add_schema(self, node: SchemaNode) -> None:
        """Add *node* to the DAG. Raises ValueError on duplicate names."""
        async with self._lock:
            if node.table.name in self._nodes:
                raise ValueError(f"Schema '{node.table.name}' already exists in the DAG.")
            self._nodes[node.table.name] = node.model_copy(deep=True)

    async def add_schemas(self, nodes: list[SchemaNode]) -> list[str]:
        """Add multiple schemas atomically.

        Returns the list of table names that were actually added (skips
        duplicates silently so that the lock-and-recheck pattern works).
        """
        added: list[str] = []
        async with self._lock:
            for node in nodes:
                if node.table.name not in self._nodes:
                    self._nodes[node.table.name] = node.model_copy(deep=True)
                    added.append(node.table.name)
        return added

    async def set_primary_key(self, table_name: str, primary_key: list[str]) -> None:
        """Set the primary key for a table after PK selection."""
        async with self._lock:
            if table_name not in self._nodes:
                raise ValueError(f"Table '{table_name}' not in DAG")
            self._nodes[table_name].primary_key = primary_key

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_snapshot(self) -> dict[str, SchemaNode]:
        """Return a deep copy of the current node map."""
        async with self._lock:
            return {k: v.model_copy(deep=True) for k, v in self._nodes.items()}

    def get_snapshot_sync(self) -> dict[str, SchemaNode]:
        """Non-async snapshot — safe only when no concurrent writers."""
        return {k: v.model_copy(deep=True) for k, v in self._nodes.items()}

    def get_root_node(self) -> SchemaNode:
        return self._nodes["DocumentMetadata"].model_copy(deep=True)

    @property
    def table_names(self) -> list[str]:
        return list(self._nodes.keys())

    def has_pk(self, table_name: str) -> bool:
        """Check whether a table exists in the DAG and has a primary key set."""
        node = self._nodes.get(table_name)
        return node is not None and node.primary_key is not None

    def __len__(self) -> int:
        return len(self._nodes)

    # ------------------------------------------------------------------
    # Prompt / string representations
    # ------------------------------------------------------------------

    def get_prompt_repr(self) -> str:
        """Format the entire DAG for inclusion in an LLM prompt.

        Produces a human-readable representation showing each table's fields,
        primary key, and foreign-key edges.
        """
        parts: list[str] = []
        for name, node in self._nodes.items():
            section = f"## Table: {name}"
            if node.table.description:
                section += f"\nDescription: {node.table.description}"
            if node.primary_key is not None:
                section += f"\nPrimary Key: {', '.join(node.primary_key)}"
            else:
                section += "\nPrimary Key: (pending)"
            if node.foreign_keys:
                fk_lines = []
                for fk in node.foreign_keys:
                    fk_lines.append(
                        f"  FK ({', '.join(fk.fields)}) -> {fk.references_table}({', '.join(fk.references_fields)})"
                    )
                section += "\nForeign Keys:\n" + "\n".join(fk_lines)
            section += "\nFields:"
            for field in node.table.fields:
                line = f"  - {field.name} ({field.data_type})"
                if field.unit:
                    line += f" [unit: {field.unit}]"
                if field.scale:
                    line += f" [scale: {field.scale}]"
                line += f": {field.description}"
                section += "\n" + line
            parts.append(section)
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Graph traversal helpers (used by selective extraction V2)
    # ------------------------------------------------------------------

    def get_forest_components(self) -> list[set[str]]:
        """Return connected components of the DAG excluding DocumentMetadata.

        Each component is a set of table names that are connected via FK
        relationships (ignoring edges to/from DocumentMetadata).  Isolated
        tables (linked only to DocumentMetadata) form their own 1-element
        component.
        """
        adj: dict[str, set[str]] = defaultdict(set)
        non_root_names: set[str] = set()

        for name, node in self._nodes.items():
            if name == "DocumentMetadata":
                continue
            non_root_names.add(name)
            for fk in node.foreign_keys:
                parent = fk.references_table
                if parent == "DocumentMetadata":
                    continue
                adj[name].add(parent)
                adj[parent].add(name)

        visited: set[str] = set()
        components: list[set[str]] = []
        for start in non_root_names:
            if start in visited:
                continue
            comp: set[str] = set()
            stack = [start]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.add(n)
                stack.extend(adj[n] - visited)
            components.append(comp)
        return components

    def get_ancestors(self, table_name: str, exclude_root: bool = True) -> set[str]:
        """All ancestor tables reachable by following FK edges upward.

        If *exclude_root* is ``True`` (default), ``DocumentMetadata`` is
        excluded from the result.
        """
        ancestors: set[str] = set()
        stack = [table_name]
        while stack:
            current = stack.pop()
            node = self._nodes.get(current)
            if node is None:
                continue
            for fk in node.foreign_keys:
                parent = fk.references_table
                if exclude_root and parent == "DocumentMetadata":
                    continue
                if parent not in ancestors:
                    ancestors.add(parent)
                    stack.append(parent)
        return ancestors

    def get_topological_order(self, table_names: set[str]) -> list[str]:
        """Topological sort of *table_names* (parents before children).

        Only FK edges between tables *within* the given set are considered.
        """
        in_degree: dict[str, int] = {n: 0 for n in table_names}
        children_map: dict[str, list[str]] = defaultdict(list)

        for name in table_names:
            node = self._nodes.get(name)
            if node is None:
                continue
            for fk in node.foreign_keys:
                parent = fk.references_table
                if parent in table_names:
                    in_degree[name] += 1
                    children_map[parent].append(name)

        queue = [n for n, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            queue.sort()
            n = queue.pop(0)
            order.append(n)
            for child in children_map.get(n, []):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)
        return order

    def get_prompt_repr_for_tables(self, table_names: set[str]) -> str:
        """Same format as :meth:`get_prompt_repr` but only for the given tables."""
        parts: list[str] = []
        for name in self._nodes:
            if name not in table_names:
                continue
            node = self._nodes[name]
            section = f"## Table: {name}"
            if node.table.description:
                section += f"\nDescription: {node.table.description}"
            if node.primary_key is not None:
                section += f"\nPrimary Key: {', '.join(node.primary_key)}"
            else:
                section += "\nPrimary Key: (pending)"
            if node.foreign_keys:
                fk_lines = []
                for fk in node.foreign_keys:
                    fk_lines.append(
                        f"  FK ({', '.join(fk.fields)}) -> {fk.references_table}({', '.join(fk.references_fields)})"
                    )
                section += "\nForeign Keys:\n" + "\n".join(fk_lines)
            section += "\nFields:"
            for field in node.table.fields:
                line = f"  - {field.name} ({field.data_type})"
                if field.unit:
                    line += f" [unit: {field.unit}]"
                if field.scale:
                    line += f" [scale: {field.scale}]"
                line += f": {field.description}"
                section += "\n" + line
            parts.append(section)
        return "\n\n".join(parts)

    def get_component_repr(self) -> str:
        """Forest-of-components representation for the Chunk Router.

        Groups non-root tables by connected component and displays each
        component with its table names, descriptions, and FK structure.
        """
        components = self.get_forest_components()
        if not components:
            return "(No tables besides DocumentMetadata)"

        parts: list[str] = []
        for idx, comp in enumerate(components, 1):
            topo = self.get_topological_order(comp)
            lines: list[str] = [f"### Component {idx}"]
            for tname in topo:
                node = self._nodes.get(tname)
                if node is None:
                    continue
                desc = node.table.description or ""
                fk_parts = []
                for fk in node.foreign_keys:
                    fk_parts.append(f"FK({', '.join(fk.fields)}) -> {fk.references_table}")
                fk_str = "; ".join(fk_parts) if fk_parts else "none"
                field_names = ", ".join(f.name for f in node.table.fields)
                lines.append(f"- **{tname}**: {desc}\n  FK: {fk_str}\n  Fields: {field_names}")
            parts.append("\n".join(lines))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_tables_model(self) -> Tables:
        """Convert the DAG into the legacy :class:`Tables` model used by
        downstream merge / answer stages.

        The ``reasoning`` field is auto-generated.
        """
        table_list = [node.table.model_copy(deep=True) for node in self._nodes.values()]
        return Tables(
            reasoning="Schema dynamically constructed during extraction.",
            tables=table_list,
        )

    def contains(self, table_name: str) -> bool:
        return table_name in self._nodes
