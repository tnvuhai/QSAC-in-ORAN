import os
import torch
from collections import deque
import torch.serialization as ts
import numpy as np

def _to_byte_tensor(x):
    """
    Ép x về torch.ByteTensor nếu có thể; nếu không trả None.
    Hỗ trợ: Tensor (auto cast), list, bytes/bytearray, numpy array (uint8).
    """
    if x is None:
        return None
    if torch.is_tensor(x):
        # Nếu không phải uint8, cố ép về uint8
        return x.to(dtype=torch.uint8, copy=False)
    if isinstance(x, (bytes, bytearray)):
        return torch.tensor(list(x), dtype=torch.uint8)
    if isinstance(x, (list, tuple)):
        return torch.tensor(x, dtype=torch.uint8)
    if isinstance(x, np.ndarray):
        # Ép dtype về uint8
        return torch.tensor(x.astype(np.uint8, copy=False), dtype=torch.uint8)
    # Không biết ép kiểu
    return None

def _set_rng_state_safe(rng_state_dict):
    """
    Đặt lại RNG state nếu hợp lệ; bỏ qua nếu dữ liệu không đúng định dạng.
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
            # cuda_states có thể là list các state; ép từng cái
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
    Chuyển replay buffer sang list các dict primitive để tránh lỗi unpickle:
    - Không lưu deque
    - Không lưu object tùy biến
    """
    if rb is None:
        return None
    # rb có thể là deque các tuple (s, a, r, ns, d) hoặc namedtuple
    out = []
    for t in list(rb):  # ép về list
        # Hỗ trợ tuple/namedtuple/dict
        if isinstance(t, dict):
            s = t.get("state"); a = t.get("action"); r = t.get("reward")
            ns = t.get("next_state"); d = t.get("done")
        else:
            # kỳ vọng thứ tự: state, action, reward, next_state, done
            s, a, r, ns, d = t
        # Ép về list/float cơ bản (tránh tensor không cần thiết)
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
            # fallback: cố gắng chuyển sang list nếu có .tolist()
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
    Nạp lại vào buffer_obj.buffer (deque/list) từ list các dict.
    """
    if serialized is None or not hasattr(buffer_obj, "buffer"):
        return
    from collections import deque as _deque
    restored = []
    for t in serialized:
        restored.append((
            t["state"], t["action"], t["reward"], t["next_state"], t["done"]
        ))
    # nếu buffer là deque, khởi tạo lại dạng deque để giữ maxlen (nếu có)
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

        # đúng tên theo SACAgent bạn đưa
        "actor_optim": agent.actor_optim.state_dict(),
        "critic_optim": agent.critic_optim.state_dict(),
        "alpha_optim": agent.alpha_optim.state_dict(),

        "log_alpha": agent.log_alpha.detach().clone(),
        "episode": episode,

        # ⬇️ lưu buffer dạng list-dict an toàn (không còn deque)
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
    # Thử cách an toàn trước: allowlist deque + weights_only=True
    try:
        with ts.safe_globals([deque]):
            ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception as e1:
        # Fallback: nếu bạn tin tưởng file, cho phép full unpickle
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            print("[WARN] Loaded with weights_only=False (trusted checkpoint).")
        except Exception as e2:
            raise e1  # ném lỗi gốc để bạn biết lý do

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