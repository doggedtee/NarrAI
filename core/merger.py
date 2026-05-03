import numpy as np
from core.embedder import embedder


def resolve_field_names(new_data: dict, existing_data: dict) -> dict:
    if not existing_data or not new_data:
        return new_data

    existing_keys = list(existing_data.keys())
    existing_embeddings = embedder.encode(existing_keys)

    resolved = {}
    for key, value in new_data.items():
        new_embedding = embedder.encode([key])
        similarities = np.dot(existing_embeddings, new_embedding[0]) / (
            np.linalg.norm(existing_embeddings, axis=1) * np.linalg.norm(new_embedding[0])
        )
        best_idx = int(np.argmax(similarities))
        if similarities[best_idx] > 0.8:
            resolved[existing_keys[best_idx]] = value
        else:
            resolved[key] = value

    return resolved


def merge_fields(existing: dict, new: dict):
    for key, value in new.items():
        if value is None or value in ("string", "list"):
            continue
        if value == "[remove]":
            existing.pop(key, None)
        elif key in existing and isinstance(existing[key], list) and isinstance(value, list):
            remove_queries = [v[len("[remove]"):].strip() for v in value if isinstance(v, str) and v.startswith("[remove]")]
            to_add = [v for v in value if not (isinstance(v, str) and v.startswith("[remove]"))]
            kept = list(existing[key])
            if remove_queries and kept:
                kept_embeddings = embedder.encode(kept)
                for query in remove_queries:
                    query_embedding = embedder.encode([query])
                    similarities = np.dot(kept_embeddings, query_embedding[0]) / (
                        np.linalg.norm(kept_embeddings, axis=1) * np.linalg.norm(query_embedding[0])
                    )
                    best_idx = int(np.argmax(similarities))
                    if similarities[best_idx] > 0.7:
                        kept.pop(best_idx)
                        kept_embeddings = np.delete(kept_embeddings, best_idx, axis=0)
            if kept and to_add:
                kept_embeddings = embedder.encode(kept)
                filtered_add = []
                for item in to_add:
                    item_embedding = embedder.encode([item])
                    similarities = np.dot(kept_embeddings, item_embedding[0]) / (
                        np.linalg.norm(kept_embeddings, axis=1) * np.linalg.norm(item_embedding[0])
                    )
                    if float(np.max(similarities)) < 0.90:
                        filtered_add.append(item)
                existing[key] = list(set(kept) | set(filtered_add))
            else:
                existing[key] = list(set(kept) | set(to_add))
        else:
            existing[key] = value


def merge_into_whole(active: dict, whole: dict) -> dict:
    if active.get("plot_threads"):
        new_plot_threads = active["plot_threads"]
        if "planned" not in new_plot_threads and "planned" in whole.get("plot_threads", {}):
            new_plot_threads["planned"] = whole["plot_threads"]["planned"]
        whole["plot_threads"] = new_plot_threads
    merge_fields(whole.setdefault("world", {}), active.get("world", {}))

    for section in ("characters", "locations"):
        for name, data in active.get(section, {}).items():
            if data.get("remove"):
                whole.get(section, {}).pop(name, None)
                continue
            real_name = data.pop("name", None)
            if real_name and real_name != name and name in whole.setdefault(section, {}):
                whole[section][real_name] = whole[section].pop(name)
                for status in ("main", "paused", "foreshadowed"):
                    for thread in whole.get("plot_threads", {}).get(status, {}).values():
                        if name in thread.get("involved", []):
                            thread["involved"] = [real_name if n == name else n for n in thread["involved"]]
                        if thread.get("current_location") == name:
                            thread["current_location"] = real_name
                name = real_name
            if name in whole.setdefault(section, {}):
                data = resolve_field_names(data, whole[section][name])
                merge_fields(whole[section][name], data)
            else:
                whole[section][name] = data

    return whole
