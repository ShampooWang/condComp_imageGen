from typing import Dict, List, Optional, Tuple, Union


def oxford_join(items: Union[str, list], article: str = "a") -> str:
    """
    Join items with an Oxford comma and add an article before each item.

    Examples:
        oxford_join(["person", "tie", "tv"]) -> "a person, a tie, and a tv"
        oxford_join(["dog", "cat"], article="the") -> "the dog and the cat"
    """
    if isinstance(items, str):
        items = [items]
    items = [str(i).strip() for i in items if i]
    if not items:
        return ""

    # prefix article
    items = [f"{article} {item}" for item in items]

    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def prepare_condComp_prompts(
    categories: list,
    add_obj: Union[str, list],
    keep_ratio_offset: int = 0,
    drop_stages: Optional[list] = None,
) -> Tuple[list, list, list]:
    if not isinstance(add_obj, list):
        add_obj = [add_obj]

    categories_list = [[cat for cat in categories if cat not in add_obj], *add_obj]
    if drop_stages is not None:
        categories_list = [
            c for i, c in enumerate(categories_list) if not drop_stages[i]
        ]
    prompts_list = [
        "A photo with realistic lighting, showing " + oxford_join(_cat) + "."
        for _cat in categories_list
    ]
    keep_ratios = [
        (
            len(_cat) / (len(categories) + keep_ratio_offset)
            if _k == 0
            else 1 / (len(categories) + keep_ratio_offset)
        )
        for _k, _cat in enumerate(categories_list)
    ]

    return prompts_list, categories_list, keep_ratios
