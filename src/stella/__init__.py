from .logging import (
    set_verbosity_info, 
    get_logger, 
    setup_root_logger
)

set_verbosity_info()

from functools import partial

from .vocab import (
    PAD_TOKEN, 
    MASK_TOKEN
)

from .utils import get_gene_symbol_vocab

from .datacollator import (
    STELLAPreCollator,
    STELLADataCollator
)

from transformers import DataCollatorWithPadding


precollator = STELLAPreCollator(pad_token=PAD_TOKEN, mask_token=MASK_TOKEN)

STELLADataCollatorV1 = DataCollatorWithPadding(precollator)
r"""
STELLADataCollatorV1: Dynamic Padding

**Examples:**
    >>> from stella import STELLADataCollatorV1
    >>> from transformers import Trainer
    >>> trainer = Trainer(
    ...     model=...,
    ...     args=...,
    ...     train_dataset=...,
    ...     data_collator=STELLADataCollatorV1
    ... )
"""

STELLADataCollatorV2 = partial(STELLADataCollator, tokenizer=precollator)
r"""
STELLADataCollatorV2: Dynamic Padding & Mask Gene Expression

**Examples:**
    >>> from stella import STELLADataCollatorV2
    >>> from transformers import Trainer
    >>> trainer = Trainer(
    ...     model=...,
    ...     args=...,
    ...     train_dataset=...,
    ...     data_collator=STELLADataCollatorV2(mlm=True, mlm_probability=...)
    ... )
"""