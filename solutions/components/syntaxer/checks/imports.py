"""Import-related bloat checks."""

from __future__ import annotations

from ..context import AnalysisContext
from ..issues import Issue, make_issue
from ..utils import node_text, iter_nodes


def run(ctx: AnalysisContext) -> list[Issue]:
    issues: list[Issue] = []
    root = ctx.tree.root_node
    
    imports_to_check = []
    
    for node in root.children:
        if node.type != "import_declaration":
            continue
        
        has_star = any(c.type in ("asterisk", "*") for c in node.children)
        if has_star:
            issues.append(
                make_issue(
                    "import_wildcard",
                    node,
                    "Wildcard import; potential library bloat.",
                )
            )
            continue
        
        import_text = node_text(node, ctx.source_bytes)
        class_name, full_path = extract_import_info(import_text)
        if class_name and full_path:
            imports_to_check.append((node, class_name, full_path))
    
    if imports_to_check:
        used_classes = find_used_classes(root, ctx.source_bytes)
        
        for import_node, class_name, full_path in imports_to_check:
            if class_name not in used_classes:
                issue = make_issue(
                    "unused_import_suspected",
                    import_node,
                    f"Import '{class_name}' appears unused (pending bytecode verification).",
                )
                issues.append(issue)
    
    return issues


def extract_import_info(import_text: str) -> tuple[str | None, str | None]:
    """Extract class name and full path from an import statement."""
    text = import_text.replace("import", "").replace(";", "").strip()
    
    is_static = False
    if text.startswith("static"):
        text = text.replace("static", "").strip()
        is_static = True
    
    parts = text.split(".")
    if not parts:
        return None, None
    
    if parts[-1].strip() == "*":
        return None, None
    
    class_name = parts[-1].strip()
    full_path = "/".join(parts)
    
    return class_name, full_path


def find_used_classes(root, source_bytes: bytes) -> set[str]:
    """
    Find all class names that are actually used in the code.
    
    Scans for:
    - Type references (variable declarations, parameters, return types)
    - Constructor calls (new ClassName())
    - Static method calls (ClassName.method())
    - Annotations (@ClassName)
    - Catch clauses (catch (ExceptionClass e))
    
    Excludes import declarations from the scan.
    """
    used = set()
    
    # Skip import declarations - only scan the actual code
    for child in root.children:
        # Skip import and package declarations
        if child.type in ("import_declaration", "package_declaration"):
            continue
        
        # Scan this part of the tree
        for node in iter_nodes(child):
            node_type = node.type
            
            if node_type in ("type_identifier", "identifier"):
                text = node_text(node, source_bytes)
                if text and text.lower() not in ("void", "var", "this", "super"):
                    used.add(text)
            
            elif node_type == "scoped_identifier":
                for grandchild in node.children:
                    if grandchild.type == "type_identifier":
                        text = node_text(grandchild, source_bytes)
                        if text:
                            used.add(text)
                        break
            
            elif node_type == "method_invocation":
                for grandchild in node.children:
                    if grandchild.type in ("type_identifier", "identifier"):
                        text = node_text(grandchild, source_bytes)
                        if text and text[0].isupper():
                            used.add(text)
                            break
    
    return used
