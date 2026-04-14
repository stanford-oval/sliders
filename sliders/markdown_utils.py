from marko.ext.gfm import gfm
from marko.ext.gfm.elements import Table as TableElement
from marko.ext.gfm.renderer import GFMRendererMixin
from marko.html_renderer import HTMLRenderer
from marko.md_renderer import MarkdownRenderer


class CustomHTMLRenderer(HTMLRenderer, GFMRendererMixin):
    pass


class CustomMarkdownRenderer(MarkdownRenderer, GFMRendererMixin):
    pass


renderer = CustomMarkdownRenderer()


def parse_markdown(markdown: str):
    return gfm.parse(markdown)


def find_table_in_markdown_doc(node, tables):
    if isinstance(node, TableElement):
        tables.append(node)
    if hasattr(node, "children"):
        for child in node.children:
            find_table_in_markdown_doc(child, tables)
