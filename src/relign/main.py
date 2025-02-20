import typer
import inquirer
from pathlib import Path

from relign.common.runtime import Runtime

DEFAULT_EXPERIMENT_DIR = "configs/experiments"

app = typer.Typer()

@app.command()
def main(
    config: str = typer.Option(None, "--config", help="Path to config file. If omitted or empty, a prompt is shown."),
    experiment_name: str = typer.Option(None, "--experiment-name", help="Optional experiment name"),
    run_name: str = typer.Option(None, "--run-name", help="Optional run name"),
    wandb_project: str = typer.Option(None, "--wandb-project", help="Optional W&B project name"),
):
    """
    Default subcommand that runs the experiment. 
    If any parameter is unset or None, we prompt inquirer.
    Additionally, if experiment_name is not set, 
    we automatically derive its value from the chosen config's filename.
    """

    # If config is None, empty string, or "None", open a menu
    if not config or config.strip().lower() == "none":
        
        config_dir = Path(DEFAULT_EXPERIMENT_DIR)
        config_files = list(config_dir.glob("*.jsonnet"))

        # If there are no jsonnet files, exit
        if not config_files:
            typer.echo("No jsonnet config files found in /configs directory.")
            raise typer.Exit()

        answers = inquirer.prompt([
            inquirer.List(
                name="config",
                message="Select a run configuration file",
                choices=[str(f.relative_to(config_dir)) for f in config_files],
            )
        ])
        if answers is None:
            typer.echo("User cancelled, exiting.")
            raise typer.Exit()

        # Construct absolute path
        chosen = answers["config"]
        config = config_dir / chosen

    else:
        # Use the provided config path
        config_path = Path(config)
        if not config_path.exists():
            typer.echo(f"Warning: provided config path {config!r} does not exist.")
        config = config_path

    # --------------------------------------------------------------------------------
    # Auto-infer the experiment name from config filename if experiment_name is empty
    # --------------------------------------------------------------------------------
    if not experiment_name:
        # If config is "my_experiment.jsonnet", use "my_experiment"
        # If there's no suffix, config.stem = config.name
        experiment_name = config.stem

    # Handle user prompt for run_name if missing
    if not run_name:
        answers = inquirer.prompt([
            inquirer.Text(
                name="run_name",
                message="Enter run name"
            )
        ])
        if answers is None:
            typer.echo("User cancelled, exiting.")
            raise typer.Exit()
        run_name = answers["run_name"]

    # Handle user prompt for wandb_project if missing
    if not wandb_project:
        answers = inquirer.prompt([
            inquirer.Text(
                name="wandb_project",
                message="Enter Weights & Biases project name"
            )
        ])
        if answers is None:
            typer.echo("User cancelled, exiting.")
            raise typer.Exit()
        wandb_project = answers["wandb_project"]

    # Initialize the runner
    runner = Runtime(
        config=config,
        experiment_name=experiment_name,
        run_name=run_name,
        wandb_project=wandb_project
    )

    typer.echo("[DEBUG] Calling runner.setup()...")
    runner.setup()

    typer.echo("[DEBUG] Calling runner.run()...")
    runner.run()

    typer.echo("[DEBUG] Calling runner.teardown()...")
    runner.teardown()

if __name__ == "__main__":
    app()