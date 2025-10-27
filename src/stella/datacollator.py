from transformers.utils import PaddingStrategy
from typing import List, Dict, Any, Union, Optional, Mapping
from transformers.tokenization_utils_base import BatchEncoding
from .vocab import PAD_TOKEN_ID, MASK_TOKEN_ID, PAD_TOKEN, MASK_TOKEN
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizerBase

EncodedInput = List[int]


# Construct bin_vocab
VOCAB = dict(zip(range(1, MASK_TOKEN_ID), range(1, MASK_TOKEN_ID)))
VOCAB[PAD_TOKEN] = PAD_TOKEN_ID
VOCAB[MASK_TOKEN] = MASK_TOKEN_ID


class STELLADataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = pad_without_fast_tokenizer_warning(
                self.tokenizer, examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
            )
        else:
            raise TypeError("The `__getitem__` function of the `STELLADataset` should return a dictionary, please check it!")

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids_gene_expression"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids_gene_expression"], special_tokens_mask=special_tokens_mask
            )
        else:
            raise NotImplementedError("Causal language model is not supported!")
        
        return batch


class STELLAPreCollator(PreTrainedTokenizerBase):
    padding_side: str = "right"
    all_special_ids: list = [PAD_TOKEN_ID, MASK_TOKEN_ID]  # `get_special_tokens_mask` needed!

    # Do not modify it!
    model_input_names: List[str] = [
        "input_ids_gene_symbol",
        "input_ids_gene_expression",
        "token_type_ids",
        "attention_mask"
    ]


    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        padding_side: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs:
                Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in `padding_side` argument:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            padding_side:
                The side on which the model should have padding applied. Should be selected between ['right', 'left'].
                Default value is picked from the class attribute of the same name.
            return_attention_mask:
                (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        required_input = encoded_inputs[self.model_input_names[0]]

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(required_input)

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD and len(required_input) != max_length

        # Initialize attention mask if not present.
        if return_attention_mask and "attention_mask" not in encoded_inputs:
            encoded_inputs["attention_mask"] = [1] * len(required_input)

        if needs_to_be_padded:
            difference = max_length - len(required_input)
            padding_side = padding_side if padding_side is not None else self.padding_side

            if padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"] + [0] * difference
                # if "token_type_ids" in encoded_inputs:
                #     encoded_inputs["token_type_ids"] = (
                #         encoded_inputs["token_type_ids"] + [self.pad_token_type_id] * difference
                #     )
                
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs[self.model_input_names[0]] = required_input + [self.pad_token_id] * difference

                # TODO: Encode "input_ids_gene_expression"
                encoded_inputs[self.model_input_names[1]] += [self.pad_token_id] * difference

            elif padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + encoded_inputs["attention_mask"]
                # if "token_type_ids" in encoded_inputs:
                #     encoded_inputs["token_type_ids"] = [self.pad_token_type_id] * difference + encoded_inputs[
                #         "token_type_ids"
                #     ]

                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs[self.model_input_names[0]] = [self.pad_token_id] * difference + required_input

                # TODO: Encode "input_ids_gene_expression"
                encoded_inputs[self.model_input_names[1]] = [self.pad_token_id] * difference + encoded_inputs[self.model_input_names[1]]

            else:
                raise ValueError(f"Invalid padding strategy:{padding_side}")

        return encoded_inputs
    

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """
        Converts a token string (or a sequence of tokens) in a single integer id (or a sequence of ids), using the
        vocabulary.

        Args:
            tokens (`str` or `List[str]`): One or several token(s) to convert to token id(s).

        Returns:
            `int` or `List[int]`: The token id or list of token ids.
        """
        if tokens is None:
            return None

        if isinstance(tokens, str):
            return self._convert_token_to_id_with_added_voc(tokens)

        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id_with_added_voc(token))
        return ids
    

    def _convert_token_to_id_with_added_voc(self, token):
        if token is None:
            return None

        return VOCAB.get(token)


    def __len__(self):
        return len(VOCAB)


if __name__ == "__main__":
    pass