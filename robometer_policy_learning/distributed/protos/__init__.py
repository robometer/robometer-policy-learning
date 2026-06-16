import importlib
import os
import sys
import re
from pathlib import Path


def _generate_protos():
    try:
        from grpc_tools import protoc
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "grpc_tools is required to generate protobufs. Install with `pip install grpcio-tools`."
        ) from e

    here = Path(__file__).parent
    out_dir = str(here)

    # Generate learner.proto
    learner_proto = str(here / "learner.proto")
    args = [
        "protoc",
        f"-I{here}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        learner_proto,
    ]
    if protoc.main(args) != 0:  # pragma: no cover
        raise RuntimeError("protoc failed to generate learner protobufs")

    # Generate reward_relabel.proto
    reward_relabel_proto = str(here / "reward_relabel.proto")
    args = [
        "protoc",
        f"-I{here}",
        f"--python_out={out_dir}",
        f"--grpc_python_out={out_dir}",
        reward_relabel_proto,
    ]
    if protoc.main(args) != 0:  # pragma: no cover
        raise RuntimeError("protoc failed to generate reward_relabel protobufs")

    # Post-process generated code for protobuf runtime compatibility (runtime_version guard)
    for pb2_name in ["learner_pb2.py", "reward_relabel_pb2.py"]:
        try:
            pb2_path = Path(out_dir) / pb2_name
            if pb2_path.exists():
                text = pb2_path.read_text()
                # 1) Make runtime_version import optional
                text = text.replace(
                    "from google.protobuf import runtime_version as _runtime_version",
                    (
                        "try:\n    from google.protobuf import runtime_version as _runtime_version\n"
                        "except Exception:\n    _runtime_version = None"
                    ),
                )
                # 2) Remove the entire ValidateProtobufRuntimeVersion(...) block if present
                text = re.sub(
                    r"_runtime_version\.ValidateProtobufRuntimeVersion\([\s\S]*?\)\n",
                    "",
                    text,
                    count=1,
                )
                pb2_path.write_text(text)
        except Exception:
            # Best-effort: continue without patching
            pass


def _ensure_generated():
    here = Path(__file__).parent

    # Check if proto files exist, generate if missing
    learner_pb2_path = here / "learner_pb2.py"
    learner_pb2_grpc_path = here / "learner_pb2_grpc.py"
    reward_relabel_pb2_path = here / "reward_relabel_pb2.py"
    reward_relabel_pb2_grpc_path = here / "reward_relabel_pb2_grpc.py"

    generated = [learner_pb2_path, learner_pb2_grpc_path, reward_relabel_pb2_path, reward_relabel_pb2_grpc_path]
    protos = [here / "learner.proto", here / "reward_relabel.proto"]

    need_generate = False
    if not all(p.exists() for p in generated):
        need_generate = True
    else:
        proto_mtime = max(p.stat().st_mtime for p in protos if p.exists())
        for gen in generated:
            if gen.stat().st_mtime < proto_mtime:
                need_generate = True
                break

    if need_generate:
        _generate_protos()

    # Make sure top-level imports inside *_pb2_grpc.py (which do `import learner_pb2`) can resolve
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    # Import top-level modules and expose them as package attrs
    try:
        lpb = importlib.import_module("learner_pb2")
        lpb_grpc = importlib.import_module("learner_pb2_grpc")
        rpb = importlib.import_module("reward_relabel_pb2")
        rpb_grpc = importlib.import_module("reward_relabel_pb2_grpc")
        globals()["learner_pb2"] = lpb
        globals()["learner_pb2_grpc"] = lpb_grpc
        globals()["reward_relabel_pb2"] = rpb
        globals()["reward_relabel_pb2_grpc"] = rpb_grpc
    except ImportError:
        # Fallback to package-qualified modules if top-level fails
        try:
            lpb = importlib.import_module("distributed.protos.learner_pb2")
            lpb_grpc = importlib.import_module("distributed.protos.learner_pb2_grpc")
            rpb = importlib.import_module("distributed.protos.reward_relabel_pb2")
            rpb_grpc = importlib.import_module("distributed.protos.reward_relabel_pb2_grpc")
        except ImportError:
            # If both fail, try generating again and retry once more
            _generate_protos()
            lpb = importlib.import_module("distributed.protos.learner_pb2")
            lpb_grpc = importlib.import_module("distributed.protos.learner_pb2_grpc")
            rpb = importlib.import_module("distributed.protos.reward_relabel_pb2")
            rpb_grpc = importlib.import_module("distributed.protos.reward_relabel_pb2_grpc")
        globals()["learner_pb2"] = lpb
        globals()["learner_pb2_grpc"] = lpb_grpc
        globals()["reward_relabel_pb2"] = rpb
        globals()["reward_relabel_pb2_grpc"] = rpb_grpc


# Set __all__ after modules are loaded
__all__ = ["learner_pb2", "learner_pb2_grpc", "reward_relabel_pb2", "reward_relabel_pb2_grpc"]


_ensure_generated()
