"""Source-balanced sampling and a fixed, scene-diverse validation panel."""

from collections import defaultdict


class Samples:
    def __init__(self, rows):
        self.sources = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for row in rows:
            self.sources[row["source"]][row["group"]][row["pair_id"]].append(row)

    def choose(self, rng):
        groups = self.sources[rng.choice(sorted(self.sources))]
        views = groups[rng.choice(sorted(groups))]
        return rng.choice(views[rng.choice(sorted(views))])


def validation_rows(rows):
    sources = defaultdict(dict)
    for row in rows:
        if row["split"] == "val":
            sources[row["source"]].setdefault(row["group"], row)
    return [groups[key] for source, groups in sorted(sources.items())
            for key in sorted(groups)[:2]]
