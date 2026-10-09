"""Source-derived eager admission, per-worker cumulative assignment, sticky routing."""


class Policy:
    def reset(self, metadata):
        self.counts = [[0, 0] for _ in range(metadata["workers"])]
        self.routes = {}

    def schedule(self, view):
        admit = [x["id"] for x in view["trajectories"] if x["state"] == "pending"]
        dispatch = []
        for x in view["trajectories"]:
            if x["state"] != "ready":
                continue
            tid = x["id"]
            if tid not in self.routes:
                counts = self.counts[x["worker"]]
                replica = min(range(2), key=lambda r: (counts[r], r))
                counts[replica] += 1
                self.routes[tid] = replica
            dispatch.append({"id": tid, "replica": self.routes[tid]})
        return {"admit": admit, "dispatch": dispatch}
