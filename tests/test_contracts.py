import torch

from ego_hand_wm.contracts.batch import canonical_collate
from ego_hand_wm.contracts.schema import ENTITY_NAMES, GEOMETRY_DIM, SCHEMA
from ego_hand_wm.data.synthetic import SyntheticCanonicalDataset


def test_schema_and_padding_masks() -> None:
    state = torch.randn(2, 3, GEOMETRY_DIM)
    split = SCHEMA.split(state)
    recovered = SCHEMA.pack(
        split["camera"],
        split["left_wrist"],
        split["right_wrist"],
        split["left_mano"],
        split["right_mano"],
    )
    torch.testing.assert_close(recovered, state)

    entities = SCHEMA.split_entities(state)
    assert len(entities) == len(ENTITY_NAMES) == 13
    torch.testing.assert_close(SCHEMA.pack_entities(entities), state)
    stream_mask = torch.tensor([[True, True, False, True, False]])
    entity_mask = SCHEMA.expand_entity_mask(stream_mask)
    assert entity_mask.shape == (1, 13)
    assert entity_mask[0, 3:8].all()
    assert not entity_mask[0, 8:].any()

    dataset = SyntheticCanonicalDataset(length=2, history_steps=3, future_steps=4, image_size=16)
    first = dataset[0]
    second = dataset[1]
    for key in (
        "future_time",
        "future_query_stream_mask",
        "future_state",
        "future_stream_mask",
    ):
        second[key] = second[key][:-1]
    batch = canonical_collate([first, second])
    assert batch.future_padding_mask.shape == (2, 4)
    assert batch.future_padding_mask[1, -1]
    assert not batch.future_padding_mask[0].any()
