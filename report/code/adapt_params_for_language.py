def _adapt_params_for_language(self, language_info: dict) -> dict:
    total = language_info["total"]
    if total == 0:
        return {}

    ru_pct = language_info["russian"] / total
    en_pct = language_info["english"] / total
    mix_pct = language_info["mixed"] / total

    params = {}

    # Vector size adaptation
    if mix_pct > 0.3 or (ru_pct > 0.2 and en_pct > 0.2):
        params["vector_size"] = min(400, self.vector_size + 50)

    # Context window adaptation
    if en_pct > 0.5:
        params["window"] = max(10, self.window - 2)
    elif mix_pct > 0.3:
        params["window"] = min(20, self.window + 3)

    # Min count adaptation
    if total < 100:
        params["min_count"] = max(1, self.min_count - 1)
    elif mix_pct > 0.3:
        params["min_count"] = self.min_count + 1

    return params
