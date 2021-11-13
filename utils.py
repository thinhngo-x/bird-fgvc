from pathlib import Path
import json
import torch


def save_ckpt_auto_rm(model, filepath, val_score, max_num=5):
    """Save checkpoint and automatically remove another checkpoint.
    Always keeping `max_num` highest val_score checkpoints.

    Args:
        model: checkpoint
        filepath:
        val_score: validation score of this checkpoint
        max_num:
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = filepath.parent / 'metadata.json'
    if metadata_path.exists():
        metadata = json.load(open(metadata_path, 'r'))
    else:
        metadata = {}
    if len(metadata)+1 > max_num:
        rm_ckpt = sorted(metadata, key=metadata.get)[0]
        Path(rm_ckpt).unlink()
        metadata.pop(rm_ckpt, None)
    metadata[str(filepath)] = val_score
    json.dump(metadata, open(metadata_path, 'w'))
    torch.save(model.state_dict(), filepath)
