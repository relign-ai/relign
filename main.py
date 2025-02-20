import typer
from pathlib import Path

from relign.common.runtime import Runtime

@typer.command()
def main(
    config: str = typer.Option(..., "--config"),
    experiment_name: str = typer.Option(..., "--experiment-name"),
    run_name: str = typer.Option(..., "--run-name"),
    wandb_project: str = typer.Option(..., "--wandb-project"),
):
    # parse the config 
    config = Path(config)

    runner = Runtime(
        config=config,
        # Parse more device information
    )

    # Call the runner setup (instantiates the classes)
    runner.setup()

    # Run the config
    runner.run()
    
    # shut evertything down
    runner.teardown()


if __name__ == "__main__":
    typer.run(main)