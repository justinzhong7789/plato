"""
Having a registry of all available wrappers is convenient for retrieving an instance
from the defined dataset for specific usages, such as constrastive learning.
"""

import logging
from collections import OrderedDict

from plato.config import Config

from plato.datasources.contrastive_data_wrapper import (
    ContrastiveDataWrapper, ContrastiveAugmentDataWrapper)

registered_datasources_wrapper = OrderedDict([
    ('ContrastiveWrapper', ContrastiveDataWrapper),
    ("ContrastiveAugmentWrapper", ContrastiveAugmentDataWrapper)
])


def get(datasource, augment_transformer=None):
    """Get the data source equipped with desired datasource_wrapper."""
    datasource_wrapper_name = Config().data.data_wrapper

    logging.info("Data source wrapper: %s", datasource_wrapper_name)

    if datasource_wrapper_name in registered_datasources_wrapper:
        obtained_wrapper = registered_datasources_wrapper[
            datasource_wrapper_name]
    else:
        raise ValueError(
            f'No such data source wrapper: {datasource_wrapper_name}')

    return obtained_wrapper(datasource, augment_transformer)
