
import sys
import os

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(parent_path)

from scripts.modules.utils.parameters_handle import get_input
from scripts.modules.dataset import DATASETS_TABLE
import scripts as kw


def main():
    '''
    This code is responsible for **preprocessing a set of datasets**. You can provide more than one dataset per execution (just respond with the dataset numbers separated by spaces).

    The preprocessing involves **removing all repeated interactions** from the datasets, specifically those with identical item, user, and interaction timestamp. In such cases, only the first interaction within a set of repeated interactions is kept, ensuring no duplicates remain in the datasets. Additionally, **all interactions with missing data are deleted**. The **column names** in the interaction tables are standardized so that all datasets can be loaded uniformly later (using the same code). Finally, a **timestamp field is generated** based on the datetime column if this information doesn't exist, and vice-versa.
    '''
    options = get_input('Choose datasets to preprocess', [
        {
            'name': 'datasets',
            'description': 'Dataset names (or indexes) to preprocess. If not provided, a interactive menu will be shown. If "all" is provided, all datasets will be preprocessed.',
            'options': DATASETS_TABLE,
            'name_column': kw.DATASET_NAME
        }
    ])[0]

    for option_index in options:
        dataset_name = DATASETS_TABLE.loc[option_index, kw.DATASET_NAME]
        print('Preprocessing {}...'.format(dataset_name))
        preprocess_function = DATASETS_TABLE.loc[option_index, kw.DATASET_PREPROCESS_FUNCTION]
        input_path = os.path.join(kw.RAW_PATH, dataset_name)
        output_path = os.path.join(kw.DATASET_PATH, dataset_name)
        preprocess_function(input_path, output_path)
        print('Preprocessing {} done!'.format(dataset_name))


if __name__ == '__main__':
    main()
    sys.exit(0)