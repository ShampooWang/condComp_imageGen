from collections import defaultdict
from typing import Dict, List, Tuple, Union

from utils.bbox import random_non_overlapping_boxes


def get_tok2id_map(model, prompt: str) -> Dict[str, list]:
    """Return a mapping from token string to list of token positions.

    Args:
        model: Model with a ``tokenizer`` attribute.
        prompt: Input prompt string.

    Returns:
        Dict mapping token strings to lists of integer positions.
    """
    tok_ids = model.tokenizer(prompt).input_ids
    tokens = model.tokenizer.convert_ids_to_tokens(tok_ids)

    tok2id = defaultdict(list)

    for i, tok in enumerate(tokens):
        tok2id[tok].append(i)

    return tok2id


def get_token_indices(
    token_map: Dict[str, int],
    categories: List[str],
    tokenizer,
    return_tokens: bool = False,
    ignore_missing: bool = False,
) -> Union[List[int], Tuple[List[int], List[str]]]:
    """
    returns a list of token indices corresponding to the given category names.

    :param token_map: dictionary mapping tokens to their indices in the prompt
    :type token_map: Dict[str, int]
    :param categories: list of category names to find token indices for
    :type categories: List[str]
    """
    token_indices = []
    target_tokens = []
    for name in categories:
        name = name.lower().strip()

        if f"{name}</w>" in token_map:
            if isinstance(token_map[f"{name}</w>"], list):
                token_indices.extend(token_map[f"{name}</w>"])
            else:
                token_indices.append(token_map[f"{name}</w>"])
            target_tokens.append(name)
        elif not ignore_missing:
            tokens = tokenizer.tokenize(name)
            tok_ids = []
            for tok in tokens:
                if isinstance(token_map[tok], list):
                    tok_ids.extend(token_map[tok])
                else:
                    tok_ids.append(token_map[tok])
            target_tokens.extend(tokens)
            token_indices.extend(tok_ids)

    if return_tokens:
        return token_indices, target_tokens

    return token_indices


def get_token_ids_and_bboxes(
    token_map: Dict[str, int],
    categories: List[str],
    img_size: int,
    tokenizer,
    return_tokens: bool = False,
    ignore_missing: bool = False,
) -> Union[
    Tuple[List[int], List[List[float]]], Tuple[List[int], List[List[float]], List[str]]
]:
    """
    returns a list of token indices and corresponding bounding boxes for the given category names.

    :param token_map: dictionary mapping tokens to their indices in the prompt
    :type token_map: Dict[str, int]
    :param categories: list of category names to find token indices for
    :type categories: List[str]
    :param bboxes: list of bounding boxes corresponding to the categories
    :type bboxes: List[List[float]]
    """
    bboxes = random_non_overlapping_boxes(
        width=img_size,
        height=img_size,
        n=len(categories),
        min_size=100,
        max_size=300,
        normalize=True,
    )

    token_indices = []
    bbox_list = []
    target_tokens = []
    for i, (name, bbox) in enumerate(zip(categories, bboxes)):
        name = name.lower().strip()

        if f"{name}</w>" in token_map:
            if isinstance(token_map[f"{name}</w>"], list):
                token_indices.extend(token_map[f"{name}</w>"])
                bbox_list.extend([bbox] * len(token_map[f"{name}</w>"]))
            else:
                token_indices.append(token_map[f"{name}</w>"])
            target_tokens.append(name)
        elif not ignore_missing:
            tokens = tokenizer.tokenize(name)
            # tok_ids = [token_map[tok] for tok in tokens]
            tok_ids = []
            for tok in tokens:
                if isinstance(token_map[tok], list):
                    tok_ids.extend(token_map[tok])
                else:
                    tok_ids.append(token_map[tok])
            target_tokens.extend(tokens)
            token_indices.extend(tok_ids)
            bbox_list.extend([bbox] * len(tok_ids))

    if return_tokens:
        return token_indices, bbox_list, target_tokens

    return token_indices, bbox_list


def get_category_token_indices(
    category: Union[str, List[str]],
    token_map: Dict[str, int],
    tokenizer,
) -> List[int]:
    if isinstance(category, str):
        category = [category]

    tok_ids = []
    for _cat in category:
        _cat = _cat.lower().strip()
        if f"{_cat}</w>" in token_map:
            ids = token_map[f"{_cat}</w>"]
            if isinstance(ids, list):
                tok_ids.extend(ids)
            else:
                tok_ids.append(ids)
        else:
            tokens = tokenizer.tokenize(_cat)
            for tok in tokens:
                if tok not in token_map:
                    continue
                ids = token_map[tok]
                if isinstance(ids, list):
                    tok_ids.extend(ids)
                else:
                    tok_ids.append(ids)

    return tok_ids
