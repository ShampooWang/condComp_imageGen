import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from diffusers.models.attention_processor import Attention


class AttentionStore:
    """Collects the cross‑attention tensors at each U‑Net layer."""

    @staticmethod
    def _blank() -> Dict[str, list]:
        return {"down": [], "mid": [], "up": []}

    def __init__(self, attn_res: Tuple[int, int], optimized_layers=None):
        self.attn_res = attn_res
        self.num_att_layers: int = -1
        self.cur_att_layer: int = 0
        self.step_store: Dict[str, list[torch.Tensor]] = self._blank()
        self.selected_step_store: Dict[str, list[torch.Tensor]] = self._blank()
        self.optimized_layers = optimized_layers
        self.attention_store: Dict[str, list[torch.Tensor]] = {}
        self.selected_attention_store: Dict[str, list[torch.Tensor]] = {}

    def __call__(self, attn: torch.Tensor, is_cross: bool, place: str):
        if is_cross and self.cur_att_layer >= 0 and place in self.step_store:
            if self.optimized_layers is not None:
                if self.cur_att_layer in self.optimized_layers:
                    self.step_store[place].append(attn.clone())
            else:
                if attn.shape[1] == math.prod(self.attn_res):
                    self.step_store[place].append(attn.clone())

        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.attention_store = self.step_store
            self.selected_attention_store = self.selected_step_store
            self.step_store = self._blank()
            self.selected_step_store = self._blank()

    def maps(self, block_type: str):
        return self.attention_store[block_type]

    def aggregate(
        self,
        where: Sequence[str],
        attention_store=None,
        attn_res=None,
        use_selected=False,
    ) -> torch.Tensor:

        if attention_store is None:
            attention_store = self.attention_store

        if use_selected:
            attention_store = self.selected_attention_store

        if attn_res is None:
            attn_res = self.attn_res

        maps = []
        num_attn_maps = 0
        for loc in where:
            if loc in attention_store:
                for m in attention_store[loc]:
                    maps.append(m.reshape(-1, attn_res[0], attn_res[1], m.shape[-1]))
                    num_attn_maps += 1
        if not maps:
            print(attention_store, attn_res)
            raise ValueError("No attention maps collected; check attn_res.")

        maps = torch.cat(maps, 0)

        return maps.sum(0) / maps.shape[0]

    def clear(self):
        self.attention_store = self._blank()
        self.step_store = self._blank()
        self.cur_att_layer = 0


class AttnProcessor:
    """Wraps the native processor, still performs attention, but stores maps."""

    def __init__(self, store: AttentionStore, place: str):
        self.store = store
        self.place = place

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, seq_len, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, b)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        enc = encoder_hidden_states if is_cross else hidden_states
        key, value = attn.to_k(enc), attn.to_v(enc)

        query, key, value = map(attn.head_to_batch_dim, (query, key, value))
        probs = attn.get_attention_scores(query, key, attention_mask)

        self.store(probs, is_cross, self.place)

        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


def register_attention_control(
    model,
    attn_res=(32, 32),
    optimized_layers: Optional[Sequence[int]] = None,
    store_original_procs=True,
) -> None:
    """Install attention processors on a model and attach an AttentionStore.

    This replaces the model's UNet attention processors with
    :class:`AttnProcessor` wrappers that capture cross-attention
    probability maps into an :class:`AttentionStore` instance. The store is
    attached to ``model.attention_store`` and the original processors are
    optionally saved on ``model.original_procs``.

    Args:
        model: Model instance that contains ``unet.attn_processors``.
        attn_res: Tuple specifying the (W_cells, H_cells) attention grid.
        optimized_layers: Optional sequence of attention layer indices to
            capture; when ``None`` all layers are recorded.
        store_original_procs: If True, saves the model's original
            processors to ``model.original_procs`` before replacing them.
    """

    if store_original_procs:
        orig = dict(model.unet.attn_processors)
        if len(orig) == 0:
            raise RuntimeError(
                "unet.attn_processors is empty; cannot save original processors."
            )
        model.original_procs = orig

    attn_store = AttentionStore(attn_res, optimized_layers)
    procs, count = {}, 0
    for name in model.unet.attn_processors.keys():
        place = (
            "mid"
            if name.startswith("mid_block")
            else "up" if name.startswith("up_blocks") else "down"
        )
        procs[name] = AttnProcessor(attn_store, place)
        count += 1
    attn_store.num_att_layers = count

    model.unet.set_attn_processor(procs)
    model.attention_store = attn_store
