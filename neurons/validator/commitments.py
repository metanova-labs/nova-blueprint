import asyncio
from types import SimpleNamespace
from typing import cast

from bittensor.core.chain_data.utils import decode_metadata


async def get_commitments(subtensor, metagraph, block_hash: str, netuid: int, min_block: int, max_block: int) -> dict:
    commits = await asyncio.gather(*[
        subtensor.substrate.query(
            module="Commitments",
            storage_function="CommitmentOf",
            params=[netuid, hotkey],
            block_hash=block_hash,
        ) for hotkey in metagraph.hotkeys
    ])

    result = {}
    for uid, hotkey in enumerate(metagraph.hotkeys):
        commit = cast(dict, commits[uid])
        if commit and min_block < commit['block'] < max_block:
            result[hotkey] = SimpleNamespace(
                uid=uid,
                hotkey=hotkey,
                block=commit['block'],
                data=decode_metadata(commit)
            )
    return result

