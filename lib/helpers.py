import json, munch, os
import numpy as np
import pandas as pd
import torch
from pathlib import Path



def load_json(folder_path, filename):
    """
    Load a JSON file from a folder, using only filename as input.
    """
    file_path = os.path.join(folder_path, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return munch.munchify(data)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON: {e}")


def load_csv(folder_path, file_name, start_row=0, end_row=None, columns_to_load=slice(None)):
    """
    Load selected rows and columns from a CSV file (skipping the header).
    Parameters
    ----------
    folder_path : str
        Path to the directory containing the CSV file.
    file_name : str
        Name of the CSV file.
    start_row : int, optional
        Starting row index (0-based, excluding header). Default is 0.
    end_row : int or None, optional
        Ending row index (exclusive). If None, loads until the last row.
    columns_to_load : slice, list, tuple, or int, optional
        Column indices to load. Default is slice(None), meaning all columns.
    Returns
    -------
    np.ndarray
        A 2D NumPy array containing the selected rows and columns.
    """

    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)

    # Check that the file exists before loading
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found in folder '{folder_path}'.")
    try:
        # Load CSV data, skip the header row, and ensure the result is always 2D
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1, ndmin=2)
        # Return the specified rows and columns
        return data[start_row:end_row, columns_to_load]
    except Exception as e:
        # Wrap any loading/slicing errors in a clearer message
        raise ValueError(f"Error while loading CSV: {e}")



def check_type(x, expected, message=None):
    """
    Check that x matches expected type(s) in python.
    x can be:
      - a scalar: float, int, complex, str, bool
      - a list or tuple of such scalars
      - a dict (if dict is expected)

    expected can be:
      - a type: float, int, complex, str, bool, list, tuple, dict
      - or a list/tuple of types, e.g. [float, int]

    message:
      None      → print nothing
      True      → same as "auto"
      str       → print that string
    """
    # Normalize expected into tuple
    if isinstance(expected, (list, tuple)):
        allowed = tuple(expected)
    else:
        allowed = (expected,)

    # Supported base types
    supported = (float, int, complex, str, bool, list, tuple, dict)

    for t in allowed:
        if not isinstance(t, type):
            raise TypeError(f"Invalid type specifier: {t}")
        if t not in supported:
            raise TypeError(f"Unsupported type: {t.__name__}")

    def is_valid_scalar(v):
        # bool must be explicitly allowed
        if isinstance(v, bool):
            return bool in allowed
        return isinstance(v, allowed)

    def print_success():
        if message is None:
            return
        if message is True:
            if isinstance(x, (list, tuple)):
                for i, v in enumerate(x):
                    print(f"check pass element {i}: type={type(v).__name__}")
            else:
                print(f"check pass: type={type(x).__name__}")
        else:
            print(message)

    # Case 1: scalar
    if is_valid_scalar(x):
        print_success()
        return

    # Case 2: list or tuple (check each element)
    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            if not is_valid_scalar(v):
                allowed_names = ", ".join(t.__name__ for t in allowed)
                raise TypeError(
                    f"Type error: The {i}-th element has type {type(v).__name__}, expected {allowed_names}"
                )
        print_success()
        return

    # Case 3: dict (do NOT check elements inside)
    if isinstance(x, dict):
        if dict in allowed:
            print_success()
            return
        allowed_names = ", ".join(t.__name__ for t in allowed)
        raise TypeError(f"Type error: expected {allowed_names}, got dict")

    # Otherwise invalid
    allowed_names = ", ".join(t.__name__ for t in allowed)
    raise TypeError(
        f"Type error: expected {allowed_names} or list/tuple of {allowed_names}, got {type(x).__name__}"
    )


def check_array(x, kind, dtype=None, shape=None, device=None, message=None):
    """
    Check array/tensor type, and optionally dtype, shape, and device.
    Parameters
    ----------
    x : array-like or list/tuple of array-like
        A numpy.ndarray, torch.Tensor, or a list/tuple of them.
    kind : str
        "numpy" or "torch" — specifies what type x should be.
    dtype : optional
        For numpy: anything accepted by np.dtype(...)
        For torch: a torch.dtype (e.g. torch.float32)
    shape : optional
        A tuple like (None, 10) where None means wildcard dimension,
        or a list of such tuples to allow multiple shapes.
    device : optional
        Torch only. e.g. "cpu", "cuda:0", "mps".
    message:
      None      → print nothing
      "true"    → print shape and dtype on success
      str       → print that string on success
    """
    kind = kind.lower().strip()
    if kind not in ("numpy", "torch"):
        raise ValueError(f"kind must be 'numpy' or 'torch', got {kind!r}")

    def check_one(a, idx=None):
        prefix = f"element {idx}: " if idx is not None else ""

        # type check
        if kind == "numpy":
            if not isinstance(a, np.ndarray):
                raise TypeError(f"{prefix}expected numpy.ndarray, got {type(a).__name__}")
        else:
            if not isinstance(a, torch.Tensor):
                raise TypeError(f"{prefix}expected torch.Tensor, got {type(a).__name__}")

        # dtype check
        if dtype is not None:
            if kind == "numpy":
                expected_dt = np.dtype(dtype)
                if a.dtype != expected_dt:
                    raise TypeError(f"{prefix}dtype mismatch: expected {expected_dt}, got {a.dtype}")
            else:
                if not isinstance(dtype, torch.dtype):
                    raise TypeError(f"{prefix}for torch, dtype must be torch.dtype, got {dtype!r}")
                if a.dtype != dtype:
                    raise TypeError(f"{prefix}dtype mismatch: expected {dtype}, got {a.dtype}")

        # shape check
        if shape is not None:
            actual = tuple(a.shape)
            specs = [shape] if isinstance(shape, tuple) else list(shape)

            def match_shape(actual, spec):
                if len(actual) != len(spec):
                    return False
                for aa, ss in zip(actual, spec):
                    if ss is None:
                        continue
                    if aa != ss:
                        return False
                return True

            if not any(match_shape(actual, spec) for spec in specs):
                raise ValueError(f"{prefix}shape mismatch: expected {specs}, got {actual}")

        # device check
        if device is not None:
            if kind != "torch":
                raise TypeError(f"{prefix}device check is only valid for kind='torch'")
            expected_dev = torch.device(device)
            if a.device != expected_dev:
                raise TypeError(f"{prefix}device mismatch: expected {expected_dev}, got {a.device}")

    # run checks
    if isinstance(x, (list, tuple)):
        for i, a in enumerate(x):
            check_one(a, idx=i)
    else:
        check_one(x)

    # success message
    if message is not None:
        if message is True:
            if isinstance(x, (list, tuple)):
                for i, a in enumerate(x):
                    print(f"check pass element {i}: shape={tuple(a.shape)}, dtype={a.dtype}")
            else:
                print(f"check pass: shape={tuple(x.shape)}, dtype={x.dtype}")
        else:
            print(message)




def numpy_to_latex(
    A: np.ndarray,
    *,
    row_names=None,
    col_names=None,
    save: bool = True,
    out_dir=".",
    filename="table.tex",
    float_format=None,          # global format for all cells (e.g. "%.4f")
    col_formats=None,           # dict: {col_name or col_index: fmt}
    row_formats=None,           # dict: {row_name or row_index: fmt}
    format_mode="global",       # "global" | "col" | "row"
    caption=None,
    label=None,
):
    """
    Convert a NumPy array into LaTeX table code, with optional saving to disk.

    Formatting:
      - format_mode="global": use float_format for all numeric cells
      - format_mode="col":    use col_formats per column
      - format_mode="row":    use row_formats per row

    Default names:
      - if row_names is None -> ["R1", ..., "Rn"]
      - if col_names is None -> ["C1", ..., "Cm"]
    """

    A = np.asarray(A)
    n_rows, n_cols = A.shape

    # 2) Default row/col names
    if row_names is None:
        row_names = [f"R{i+1}" for i in range(n_rows)]
    if col_names is None:
        col_names = [f"C{j+1}" for j in range(n_cols)]

    df = pd.DataFrame(A, index=row_names, columns=col_names)

    # Helper: apply either "%.3f" or "{:.3f}"
    def fmt_one(x, fmt):
        if callable(fmt):
            return fmt(x)
        if "%" in fmt:
            return fmt % x
        return fmt.format(x)

    # 1) Formatting choice
    if format_mode == "col":
        if col_formats is None:
            raise ValueError("format_mode='col' requires col_formats.")
        df = df.copy()
        for key, fmt in col_formats.items():
            col = df.columns[key] if isinstance(key, int) else key
            df[col] = df[col].map(lambda x, f=fmt: fmt_one(x, f))
        float_format = None  # already converted to strings

    elif format_mode == "row":
        if row_formats is None:
            raise ValueError("format_mode='row' requires row_formats.")
        df = df.copy()
        for key, fmt in row_formats.items():
            row = df.index[key] if isinstance(key, int) else key
            df.loc[row, :] = df.loc[row, :].map(lambda x, f=fmt: fmt_one(x, f))
        float_format = None  # already converted to strings

    elif format_mode != "global":
        raise ValueError("format_mode must be one of: 'global', 'col', 'row'.")

    latex = df.to_latex(
        index=True,     # now always true because we always have default row names
        header=True,    # now always true because we always have default col names
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
    )

    if save:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / filename).write_text(latex, encoding="utf-8")

    return latex




# def numpy_to_latex1(
#     A: np.ndarray,
#     *,
#     row_names=None,
#     col_names=None,
#     save: bool = True,
#     out_dir=".",
#     filename="table.tex",
#     float_format=None,
#     caption=None,
#     label=None,
# ):
#     """
#     Convert a NumPy array into LaTeX table code, with optional saving to disk.
#
#     Parameters
#     ----------
#     A : np.ndarray
#         2D NumPy array of shape (n_rows, n_cols).
#
#     row_names : list of str or None, optional
#         Row labels. If None, row indices are omitted.
#
#     col_names : list of str or None, optional
#         Column labels. If None, column headers are omitted.
#
#     save : bool, optional
#         Whether to save the LaTeX code to disk.
#
#     out_dir : str or pathlib.Path, optional
#         Directory where the LaTeX file will be saved.
#         Ignored if `save=False`.
#
#     filename : str, optional
#         Name of the LaTeX file (e.g. "results.tex").
#         Ignored if `save=False`.
#
#     float_format : str or None, optional
#         Float formatting passed to pandas `to_latex`.
#         Use None to display full numbers.
#
#     caption : str or None, optional
#         LaTeX table caption.
#
#     label : str or None, optional
#         LaTeX label for cross-referencing.
#
#     Returns
#     -------
#     latex : str
#         Generated LaTeX code.
#     """
#
#     df = pd.DataFrame(
#         A,
#         index=row_names,
#         columns=col_names
#     )
#
#     latex = df.to_latex(
#         index=row_names is not None,
#         header=col_names is not None,
#         float_format=float_format,
#         caption=caption,
#         label=label,
#         escape=False
#     )
#
#     if save:
#         out_dir = Path(out_dir)
#         out_dir.mkdir(parents=True, exist_ok=True)
#         out_path = out_dir / filename
#         out_path.write_text(latex, encoding="utf-8")
#
#     return latex




