
import pandas as pd
import argparse

from typing import TypedDict

class InputInfo(TypedDict):
    name: str
    description: str
    options: pd.DataFrame
    name_column: str


def _display_options(options_table: pd.DataFrame, name_column: str):
    '''
    Prints the available options for the user to choose from.

    params:
        options_table: Table with the available options (must have the name_column)
        name_column: Name of the column that contains the option name
    '''
    for idx, row in options_table.iterrows():        
        print('[{}] {}\n'.format(idx, row.loc[name_column]))


def ask_options(options_name: str, options_table: pd.DataFrame, name_column: str) -> 'list[int]':
    '''
    Asks and collects the options chosen by the user.

    params:
        options_name: Name of what the user is choosing (e.g., 'algorithm', 'dataset', etc.)
        options_table: Table with the available options (must have the name_column). Each row of the table is an option.
        name_column: Name of the column that contains the option name

    return:
        List of integers with the options chosen by the user
    '''
    print('\nAvailable {}:\n'.format(options_name))
    _display_options(options_table, name_column)
    options = input('Select which {} to execute: '.format(options_name))
    if options.strip() == '':
        return []
    return list(map(int, options.split(' ')))

def get_input(description: str, inputs_info: 'list[InputInfo]') -> 'list[list[int]]':
    '''
    Collects the options chosen by the user (through input or command line)

    params:
        description: Description of the options collection
        inputs_info: List of dictionaries with the following information:
            - name: Name of what the user is choosing (e.g., 'algorithm', 'dataset', etc.)
            - description: Description of what the user is choosing (e.g., 'Choose the algorithm to execute')
            - name_column: Name of the column that contains the name
            - options: Table with the available options (must have the name_column). Each row of the table is an option.

    return:
        List of lists of integers, where each list of integers represents the options chosen by the user for an InputInfo
    '''
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for input_info in inputs_info:
        parser.add_argument('--{}'.format(input_info['name']), type=str, default=None, help=input_info['description'])

    args = vars(parser.parse_args())

    options = []
    for input_info in inputs_info:

        current_options = []
        current_arg = args[input_info['name']]

        if current_arg is None:
            current_options = ask_options(input_info['name'], input_info['options'], input_info['name_column'])
        elif current_arg == 'all':
            current_options = input_info['options'].index.tolist()
        elif current_arg.replace(" ", "").replace(',', '').isdigit():
            current_options = []
            current_arg = current_arg.split(',')
            for option in current_arg:
                if int(option) not in input_info['options'].index:
                    raise ValueError('{} index {} not found!'.format(input_info['name'], option))
                current_options.append(int(option))
        else:
            current_options = []
            options_names = current_arg.split(',')
            for option_name in options_names:
                if not input_info['options'][input_info['name_column']].str.contains(option_name).any():
                    raise ValueError('{} {} not found!'.format(input_info['name'], option_name))
                current_options.append(input_info['options'][input_info['options'][input_info['name_column']].str.contains(option_name)].index.tolist()[0])

        options.append(list(current_options))

    return options