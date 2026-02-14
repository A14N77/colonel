"""Config command: manage colonel settings."""

from __future__ import annotations

import typer

from colonel.utils.rich_output import (
    console,
    print_error,
    print_info,
    print_success,
)

config_app = typer.Typer(no_args_is_help=True)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(help="Config key (e.g. 'anthropic_api_key')."),
    value: str = typer.Argument(help="Config value."),
) -> None:
    """Set a configuration value.

    Saves to .colonel/config.json in the current directory.

    Examples:
        colonel config set anthropic_api_key sk-ant-...
        colonel config set default_evaluator ncu
        colonel config set anthropic_model claude-sonnet-4-20250514
    """
    from colonel.config.settings import ColonelSettings, save_config

    # Validate the key is a known setting
    known_keys = set(ColonelSettings.model_fields.keys())
    if key not in known_keys:
        print_error(
            f"Unknown config key '{key}'. "
            f"Known keys: {', '.join(sorted(known_keys))}"
        )
        raise typer.Exit(1)

    # Mask sensitive values in output
    display_value = value
    if "key" in key.lower() or "secret" in key.lower():
        display_value = value[:8] + "..." if len(value) > 8 else "***"

    path = save_config(key, value)
    print_success(f"Set {key} = {display_value}")
    print_info(f"Saved to {path}")


@config_app.command("show")
def config_show() -> None:
    """Show current configuration.

    Displays all settings with their current values and sources.
    Sensitive values (API keys) are masked.

    Examples:
        colonel config show
    """
    from rich.table import Table

    from colonel.config.settings import get_settings

    settings = get_settings()

    table = Table(title="Colonel Configuration", show_header=True)
    table.add_column("Key", style="cyan")
    table.add_column("Value", style="bright_white")
    table.add_column("Default", style="dim")

    defaults = ColonelSettings()

    for field_name, field_info in ColonelSettings.model_fields.items():
        value = getattr(settings, field_name)
        default = getattr(defaults, field_name)

        # Mask sensitive values
        display_value = str(value)
        if ("key" in field_name.lower() or "secret" in field_name.lower()) and value:
            display_value = str(value)[:8] + "..." if len(str(value)) > 8 else "***"

        is_default = value == default
        default_str = str(default)
        if "key" in field_name.lower() and default:
            default_str = "***"

        style = "dim" if is_default else "bright_white"
        table.add_row(field_name, f"[{style}]{display_value}[/{style}]", default_str)

    console.print(table)


@config_app.command("path")
def config_path() -> None:
    """Show the configuration file path.

    Examples:
        colonel config path
    """
    from colonel.config.settings import get_config_dir

    config_dir = get_config_dir()
    config_file = config_dir / "config.json"
    print_info(f"Config directory: {config_dir}")
    print_info(f"Config file: {config_file}")
    if config_file.is_file():
        print_success("Config file exists.")
    else:
        print_info("Config file does not exist yet. Run `colonel config set` to create it.")


# Need to import ColonelSettings at module level for config_show
from colonel.config.settings import ColonelSettings  # noqa: E402
