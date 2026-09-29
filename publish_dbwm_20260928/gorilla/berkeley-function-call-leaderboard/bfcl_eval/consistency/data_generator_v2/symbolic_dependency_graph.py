'''State-path relationships and success-branch dependency graph.'''

from __future__ import annotations

import re
from dataclasses import dataclass


def segments(path):
    path = re.sub(r'^\$\.state_(?:before|after)', '', path)
    return [(dot or bracket).strip(chr(39) + chr(34)) for dot, bracket in re.findall(
        r'\.([\w]+)|\[(\*|\d+|[^\]]+)\]', path)]


def covers(parent, child):
    '''Parent and wildcard state paths cover descendants, not ancestors.'''
    left, right = segments(parent), segments(child)
    return len(left) <= len(right) and all(
        a == b or a == '*' or b == '*' or a.startswith('{') or b.startswith('{')
        for a, b in zip(left, right))


def overlaps(a, b):
    return covers(a, b) or covers(b, a)


@dataclass(frozen=True)
class Node:
    tool: str
    branch: str
    returns: tuple
    writes: tuple

    @property
    def sources(self):
        return tuple(dict.fromkeys(relation['path']
            for item in (*self.returns, *self.writes)
            for relation in item['state_source_relations']))


class DependencyGraph:
    def __init__(self, specs):
        self.nodes = []
        for spec in specs.values():
            if spec['schema_version'] not in {'1.1', '1.2'}:
                raise ValueError('Rule planning requires complete 1.1 or 1.2 source relations')
            for branch in spec['branches']:
                name = branch['id']
                if name.startswith('success_'):
                    self.nodes.append(Node(spec['tool'], name,
                        tuple(field for result in spec['returns'] if result['branch'] == name
                              for field in result['fields']),
                        tuple(item for item in spec['mutations'] if item['branch'] == name)))

    def writers(self, path):
        return [node for node in self.nodes if any(overlaps(m['target'], path) for m in node.writes)]
