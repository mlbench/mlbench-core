# -*- coding: utf-8 -*-

"""Console script for mlbench_cli."""
from mlbench_core.api import ApiClient, MLBENCH_IMAGES

import click
from tabulate import tabulate

import sys


@click.group()
def cli(args=None):
    """Console script for mlbench_cli."""
    return 0


@cli.command()
@click.argument('name', type=str)
@click.argument('num_workers', nargs=-1, type=int)
@click.option('--dashboard-url', '--u', default=None, type=str)
def run(name, num_workers, dashboard_url):
    """Start a new run for a benchmark image"""
    images = list(MLBENCH_IMAGES.keys())

    text_prompt = 'Benchmark: \n\n'

    text_prompt += '\n'.join(
        '[{}]\t{}'.format(i, t) for i, t in enumerate(images)
    )
    text_prompt += '\n[{}]\tCustom Image'.format(len(images))

    text_prompt += '\n\nSelection'

    selection = click.prompt(
        text_prompt,
        type=click.IntRange(0, len(images)),
        default=0
    )

    if selection == len(images):
        # run custom image
        image = click.prompt('Image:', type=str)
        image_command = click.prompt('Command:', type=str)
        run_on_all = click.confirm(
            'Run command on all nodes (otherwise just first node):', type=bool)
        benchmark = {
            "custom_image_name": image,
            "custom_image_command": image_command,
            "custom_image_all_nodes": run_on_all
        }
    else:
        benchmark = {"image": images[selection]}

    client = ApiClient(in_cluster=False, url=dashboard_url)

    results = []

    for num_w in num_workers:
        current_name = '{}-{}'.format(name, num_w)

        res = client.create_run(current_name, num_w, **benchmark)
        results.append(res)

    for res in results:
        act_result = res.result()
        if act_result.status > 201:
            click.Abort("Couldn't start run: {}".format(act_result.json()["message"]))
        click.echo("Run started with name {}".format(act_result.json()["name"]))


@cli.command()
@click.argument('name', type=str)
@click.option('--dashboard-url', '--u', default=None, type=str)
def status(name, dashboard_url):
    """Get the status of a benchmark run"""
    client = ApiClient(in_cluster=False, url=dashboard_url)

    ret = client.get_runs()
    runs = ret.result().json()

    run = next(r for r in runs if r['name'] == name)

    del run["job_id"]
    del run["job_metadata"]

    click.echo(tabulate([run], headers="keys"))


@cli.command()
@click.argument('name', type=str)
@click.option('--output', '-o', type=str)
@click.option('--dashboard-url', '--u', default=None, type=str)
def download(name, output, dashboard_url):
    """Download the results of a benchmark run"""
    client = ApiClient(in_cluster=False, url=dashboard_url)

    ret = client.get_runs()
    runs = ret.result().json()

    run = next(r for r in runs if r['name'] == name)

    ret = client.download_run_metrics(run['id'])

    with open(output, 'wb') as f:
        f.write(ret.result().content)


if __name__ == "__main__":
    sys.exit(cli())  # pragma: no cover
