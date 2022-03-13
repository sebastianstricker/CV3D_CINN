def parse_config(config, field, extra_fields):
    assert field in config.keys(), f"Missing field {field} in config.json."
    parsed_config = config[field]

    for extra_field in extra_fields:
        assert extra_field in list(config.keys()), f"Missing extra field {extra_field} in config.json"
        parsed_config[extra_field] = config[extra_field]

    return parsed_config
