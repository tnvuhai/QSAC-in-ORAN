import os
import torch
from collections import deque
import torch.serialization as ts
import numpy as np

def _to_byte_tensor(x):
    """
    Cast x to torch.ByteTensor if possible; otherwise return None.
    Supports: Tensor (auto cast), list, bytes/bytearray, numpy array (uint8).
    """
    if x is None:
        return None
    if torch.is_tensor(x):
        # If not uint8, try to cast to uint8
        return x.to(dtype=torch.uint8, copy=False)
    if isinstance(x, (bytes, bytearray)):
        return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.uint8)
    if isinstance(x, np.ndarray):
        # Cast dtype to uint8
        return torch.tensor(x.astype(np.uint8, copy=False), dtype=torch.uint8)
    # Unknown type to cast
    return None

def _set_rng_state_safe(rng_state_dict):
    """
    Reset RNG state if valid; skip if data is not in the correct format.
    rng_state_dict: {"torch": <...>, "cuda": <list of ...> or None}
    """
    if not rng_state_dict:
        return
    try:
        ts = _to_byte_tensor(rng_state_dict.get("torch"))
        if ts is not None:
            torch.set_rng_state(ts)
        else:
            print("[WARN] Skip setting CPU RNG: format not recognized.")
    except Exception as e:
        print(f"[WARN] Failed to set CPU RNG: {e}")

    try:
        cuda_states = rng_state_dict.get("cuda")
        if cuda_states is not None and torch.cuda.is_available():
            # cuda_states can be a list of states; cast each one
            fixed = []
            for s in cuda_states:
                bt = _to_byte_tensor(s)
                if bt is None:
                    print("[WARN] Skip one CUDA RNG: format not recognized.")
                    return
                fixed.append(bt)
            torch.cuda.set_rng_state_all(fixed)
    except Exception as e:
        print(f"[WARN] Failed to set CUDA RNG: {e}")

def _to_serializable_buffer(rb):
    """
    Convert replay buffer to a list of primitive dicts to avoid unpickling errors:
    - Do not save deque
    - Do not save custom objects
    """
    if rb is None:
        return None
    # rb can be a deque of tuples (s, a, r, ns, d) or namedtuples
    out = []
    for t in list(rb):  # cast to list
        # Support tuple/namedtuple/dict
        if isinstance(t, dict):
            s = t.get("state"); a = t.get("action"); r = t.get("reward")
            ns = t.get("next_state"); d = t.get("done")
        else:
            # expected order: state, action, reward, next_state, done
            s, a, r, ns, d = t
        # Cast to basic list/float (avoid unnecessary tensors)
        def to_basic(x):
            import numpy as np
            if torch.is_tensor(x):
                return x.detach().cpu().numpy().tolist()
            if isinstance(x, (np.ndarray,)):
                return x.tolist()
            if isinstance(x, (list, tuple)):
                return [to_basic(xx) for xx in x]
            if isinstance(x, (float, int, bool)) or x is None:
                return x
            # fallback: try to convert to list if .tolist() exists
            return x
        out.append({
            "state": to_basic(s),
            "action": to_basic(a),
            "reward": float(r),
            "next_state": to_basic(ns),
            "done": bool(d),
        })
    return out

def _restore_buffer(serialized, buffer_obj):
    """
    Reload into buffer_obj.buffer (deque/list) from a list of dicts.
    """
    if serialized is None or not hasattr(buffer_obj, "buffer"):
        return
    from collections import deque as _deque
    restored = []
    for t in serialized:
        restored.append((
            t["state"], t["action"], t["reward"], t["next_state"], t["done"]
        ))
    # if buffer is a deque, re-initialize as a deque to preserve maxlen (if any)
    if isinstance(buffer_obj.buffer, _deque) and hasattr(buffer_obj.buffer, "maxlen"):
        buffer_obj.buffer = _deque(restored, maxlen=buffer_obj.buffer.maxlen)
    else:
        buffer_obj.buffer = restored

def save_checkpoint(agent, buffer, episode, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "actor": agent.actor.state_dict(),
        "critic": agent.critic.state_dict(),
        "critic_target": agent.critic_target.state_dict(),

        # correct names according to the SACAgent provided
        "actor_optim": agent.actor_optim.state_dict(),
        "critic_optim": agent.critic_optim.state_dict(),
        "alpha_optim": agent.alpha_optim.state_dict(),

        "log_alpha": agent.log_alpha.detach().clone(),
        "episode": episode,

        # save buffer as a safe list-dict (no longer a deque)
        "replay_buffer": _to_serializable_buffer(getattr(buffer, "buffer", None)),

        "hparams": {
            "max_action": getattr(agent, "max_action", None),
            "target_entropy": getattr(agent, "target_entropy", None),
            "n_qubits": getattr(agent.actor, "n_qubits", None) if hasattr(agent.actor, "n_qubits") else None,
            "qnn_layers": getattr(agent.actor, "qnn_layers", None) if hasattr(agent.actor, "qnn_layers") else None,
        },
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }

    torch.save(ckpt, path)
    print(f"[INFO] Saved checkpoint at episode {episode} → {path}")

def load_checkpoint(agent, buffer, path, device="cpu"):
    # Try the safe way first: allowlist deque + weights_only=True
    try:
        with ts.safe_globals([deque]):
            ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception as e1:
        # Fallback: if you trust the file, allow full unpickling
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            print("[WARN] Loaded with weights_only=False (trusted checkpoint).")
        except Exception as e2:
            raise e1  # throw the original error so you know the reason

    # --- models ---
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    if "critic_target" in ckpt and hasattr(agent, "critic_target"):
        agent.critic_target.load_state_dict(ckpt["critic_target"])

    # --- log_alpha ---
    if "log_alpha" in ckpt and hasattr(agent, "log_alpha") and ckpt["log_alpha"] is not None:
        agent.log_alpha.data = ckpt["log_alpha"].to(device).data
        agent.log_alpha.requires_grad_(True)

    # --- optimizers ---
    if "actor_optim" in ckpt and hasattr(agent, "actor_optim"):
        agent.actor_optim.load_state_dict(ckpt["actor_optim"])
    if "critic_optim" in ckpt and hasattr(agent, "critic_optim"):
        agent.critic_optim.load_state_dict(ckpt["critic_optim"])
    if "alpha_optim" in ckpt and hasattr(agent, "alpha_optim"):
        agent.alpha_optim.load_state_dict(ckpt["alpha_optim"])

    # --- replay buffer ---
    if "replay_buffer" in ckpt:
        _restore_buffer(ckpt["replay_buffer"], buffer)
    else:
        print("[WARN] Replay buffer not in checkpoint.")

    # --- RNG ---
    rng = ckpt.get("rng_state", None)
    _set_rng_state_safe(rng)

    start_ep = ckpt.get("episode", -1) + 1
    print(f"[INFO] Loaded checkpoint from {path} (resume at episode {start_ep})")
    return start_ep


def load_actor_from_checkpoint(path: str, actor: torch.nn.Module, map_location=None):
    """
    Robust loader:
      - Supports full dict checkpoints {'actor': state_dict, ...} or direct state_dict
      - Automatically tries common keys: 'actor', 'state_dict', 'policy'
      - Prints missing/unexpected keys for debugging
    """
    if map_location is None:
        map_location = torch.device("cpu")
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)

    # Tìm state_dict của actor
    candidate = None
    if isinstance(ckpt, dict):
        if "actor" in ckpt and isinstance(ckpt["actor"], (dict,)):
            candidate = ckpt["actor"]
        elif "state_dict" in ckpt and isinstance(ckpt["state_dict"], (dict,)):
            candidate = ckpt["state_dict"]
        elif "policy" in ckpt and isinstance(ckpt["policy"], (dict,)):
            candidate = ckpt["policy"]
        else:
            # If it's a dict but the key is unclear, it might be the state_dict itself
            candidate = ckpt
    else:
        candidate = ckpt

    missing, unexpected = actor.load_state_dict(candidate, strict=True)
    if missing:
        print(f"[WARN] Missing keys in actor: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in actor: {unexpected}")
    actor.eval()

    # Return any metadata from the ckpt (if available) to write to the info file
    meta = {}
    if isinstance(ckpt, dict):
        for k in ["arch", "config", "meta", "epoch", "step", "train_args"]:
            if k in ckpt:
                meta[k] = ckpt[k]
    return meta