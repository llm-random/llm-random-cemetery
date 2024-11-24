import neptune as neptune
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker

def get_neptune_table(tags, negative_tags=None, columns=None):
    """
    Fetches a Neptune runs table filtered by tags and returns it as a pandas DataFrame.

    Parameters:
    - tags (list): List of tags to filter the runs.
    - negative_tags (list, optional): List of tags to exclude from the runs.
    - columns (list, optional): Additional columns to include in the runs table.

    Returns:
    - pandas.DataFrame: The runs table with the specified filters and columns.
    """

    # Initialize the Neptune project
    project = neptune.init_project(
        project="pmtest/llm-random",
        mode="read-only",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMDY0ZDI5Ni05YWU3LTQyNGYtYmY4My1hZTFkY2EzYmUwMjgifQ=="
    )

    # Fetch the runs table with the specified tags and columns
    runs_table = project.fetch_runs_table(
        tag=tags,
        columns=columns
    ).to_pandas()

    # Ensure 'sys/tags' is a list for each run
    runs_table['sys/tags'] = runs_table['sys/tags'].apply(
        lambda x: x.split(',') if isinstance(x, str) else x
    )

    # Exclude runs containing any of the negative tags
    if negative_tags:
        for neg_tag in negative_tags:
            runs_table = runs_table[
                ~runs_table['sys/tags'].apply(lambda x: neg_tag in x)
            ]

    print(f'Table downloaded\nShape: {runs_table.shape}')
    return runs_table


# TODO
# change to activation norm data
def get_activations(runs_table, metric=None):
    activation_dict = {}
    for _, run_row in runs_table.iterrows():
        run_id = run_row['sys/id']  # Assuming 'sys/id' is the run identifier
        print(f'run ID: {run_id}')
        project_name = "pmtest/llm-random"
        # run_id = "LLMRANDOM-2078"
        run = neptune.init_run(
            project=project_name,
            with_id=run_id,
            mode="read-only",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyMDY0ZDI5Ni05YWU3LTQyNGYtYmY4My1hZTFkY2EzYmUwMjgifQ==",
        )
        print(f'run keys\n{run["block_0/residual_attention"]}')
        single_run_data = {}
        print(f'n_blocks: {run["args/n_blocks"].fetch()}')
        for i in range(run_row['args/n_blocks']):
            single_run_data[i] = {
                'attn': run[f'block_{i}/residual_attention/update_norms/{metric}'].fetch_values(),
                'FF': run[f'block_{i}/residual_feedforward/update_norms/{metric}'].fetch_values(),
            }
        activation_dict[run_id] = single_run_data
    return activation_dict

def plot_module(values_dict, module_keyword, layer_num):
    """
    Plots the activation values of a specific module in a specific layer across different runs over training steps.

    Parameters:
    - values_dict (dict): The dictionary containing the activation data from get_activations().
    - module_keyword (str): 'FF' or 'attn', specifying which module to plot.
    - layer_num (int): The specific layer (block) number to plot.

    Returns:
    - None
    """

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Iterate over each run
    for run_id, run_data in values_dict.items():
        # Check if the run has the specified layer
        if layer_num in run_data:
            block_data = run_data[layer_num]
            module_data = block_data.get(module_keyword)
            if module_data is not None and not module_data.empty:
                # Ensure module_data is a DataFrame with 'step' and 'value' columns
                steps = module_data['step'].to_numpy()
                values = module_data['value'].to_numpy()
                plt.plot(steps, values, label=run_id)
            else:
                print(f"Data for run {run_id}, block {layer_num}, module {module_keyword} is missing or empty.")
        else:
            print(f"Run {run_id} does not have block {layer_num}.")

    plt.xlabel('Training Step')
    plt.ylabel('Activation Value')
    plt.title(f'Activation of {module_keyword} in Layer {layer_num}')
    plt.legend()
    plt.grid(True)
    plt.show()
