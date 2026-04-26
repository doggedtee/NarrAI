import numpy as np
from core.embedder import embedder


def get_schema(whole_state: dict) -> dict:
    plot_threads = whole_state.get("plot_threads", {})

    main_chars = set()
    for thread in plot_threads.get("main", {}).values():
        main_chars.update(thread.get("involved", []))

    secondary_chars = set()
    for status in ("paused", "foreshadowed"):
        for thread in plot_threads.get(status, {}).values():
            secondary_chars.update(thread.get("involved", []))
    secondary_chars -= main_chars

    main_location = None
    main_thread = next(iter(plot_threads.get("main", {}).values()), None)
    if main_thread:
        main_location = main_thread.get("current_location")

    schema = {
        "plot_threads": plot_threads,
        "world": {k: "list" if isinstance(v, list) else "string" for k, v in whole_state.get("world", {}).items()} or {},
        "characters": {},
        "locations": {}
    }

    all_chars = list(whole_state.get("characters", {}).keys())
    all_locs = list(whole_state.get("locations", {}).keys())

    def fuzzy_match(name, candidates):
        if not candidates:
            return None
        if name in candidates:
            return name
        emb_name = embedder.encode([name])
        emb_cands = embedder.encode(candidates)
        sims = np.dot(emb_cands, emb_name[0]) / (np.linalg.norm(emb_cands, axis=1) * np.linalg.norm(emb_name[0]))
        best = int(np.argmax(sims))
        return candidates[best] if sims[best] > 0.8 else None

    resolved_main_chars = {fuzzy_match(n, all_chars) for n in main_chars}
    resolved_main_chars.discard(None)
    resolved_secondary_chars = {fuzzy_match(n, all_chars) for n in secondary_chars}
    resolved_secondary_chars.discard(None)
    resolved_secondary_chars -= resolved_main_chars

    resolved_main_location = fuzzy_match(main_location, all_locs) if main_location else None

    for name, data in whole_state.get("characters", {}).items():
        if name in resolved_main_chars and isinstance(data, dict):
            schema["characters"][name] = {k: "list" if isinstance(v, list) else "string" for k, v in data.items()}
        elif name in resolved_secondary_chars:
            schema["characters"][name] = {}

    for name, data in whole_state.get("locations", {}).items():
        if name == resolved_main_location and isinstance(data, dict):
            schema["locations"][name] = {k: "list" if isinstance(v, list) else "string" for k, v in data.items()}
        else:
            schema["locations"][name] = {}

    return schema


def select_context(whole_state: dict) -> dict:
    plot_threads = whole_state.get("plot_threads", {})
    involved_names = set()
    for thread in plot_threads.get("main", {}).values():
        involved_names.update(thread.get("involved", []))

    context = {"plot_threads": whole_state.get("plot_threads", {}), "world": whole_state.get("world", {}), "characters": {}, "locations": {}}
    for name, data in whole_state.get("characters", {}).items():
        if name in involved_names:
            context["characters"][name] = data

    main_thread = next(iter(plot_threads.get("main", {}).values()), None)
    main_location = main_thread.get("current_location") if main_thread else None
    if main_location and main_location in whole_state.get("locations", {}):
        context["locations"][main_location] = whole_state["locations"][main_location]

    return context
