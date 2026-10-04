"""Bounded candidate-only failure details; never format runtime values or frames."""
PATH = "/workspace/candidate/policy.py"
PHASES = {"compile", "module", "Policy.__init__", "Policy.reset", "Policy.schedule",
          "Policy.schedule.result"}
HINTS = {
    "AttributeError": "initialize the referenced attribute before using it",
    "NameError": "define the referenced name before using it",
    "UnboundLocalError": "assign the local variable on every path before using it",
    "KeyError": "check that the required mapping key exists before lookup",
    "IndexError": "check sequence bounds before indexing",
    "ZeroDivisionError": "guard a zero denominator before division or modulo",
    "TypeError": "check operand types, call arguments, and required method signatures",
    "ValueError": "check that the value satisfies the called operation's requirements",
    "OverflowError": "bound numeric magnitudes before conversion or arithmetic",
    "RecursionError": "bound recursion and avoid cyclic result structures",
    "MemoryError": "reduce memory use below the 512 MiB worker limit",
    "ImportError": "use permitted preloaded helpers without filesystem imports",
    "ModuleNotFoundError": "use permitted preloaded helpers without filesystem imports",
    "AssertionError": "repair the failed candidate assertion at the reported location",
    "SyntaxError": "repair the source syntax at the reported line and column",
    "IndentationError": "repair indentation at the reported line and column",
    "TabError": "use consistent indentation at the reported line and column",
    "UnicodeDecodeError": "save the candidate source as valid UTF-8",
    "StopIteration": "handle exhausted iterators within the callback",
    "SystemExit": "return normally instead of exiting the callback worker",
    "RuntimeError": "repair the candidate runtime condition at the reported location",
    "Exception": "repair the raised candidate exception at the reported location",
}


def bounded_text(value, limit):
    return "".join(c if c.isprintable() else " " for c in value[:limit])


def sanitize_detail(value):
    """Revalidate the bounded IPC detail; arbitrary keys/tracebacks are discarded."""
    if (not isinstance(value, dict) or not isinstance(value.get("phase"), str)
            or value["phase"] not in PHASES):
        return {}
    detail = {"phase": value["phase"]}
    name = value.get("exception_type")
    if isinstance(name, str) and len(name) <= 80 and name.isascii() and name.isidentifier():
        detail["exception_type"] = name
    for key in ("line", "column"):
        number = value.get(key)
        if type(number) is int and 1 <= number <= 1048576:
            detail[key] = number
    # Only the compiler sees source text here, never formal metadata. Callback
    # exception messages can contain opaque IDs/values and are not forwarded.
    reason = value.get("reason")
    if detail["phase"] == "compile" and isinstance(reason, str):
        detail["reason"] = bounded_text(reason, 256)
    return detail


def exception_detail(exc, phase):
    detail = {"phase": phase, "exception_type": type(exc).__name__}
    frame = exc.__traceback__
    for _ in range(128):
        if frame is None:
            break
        if frame.tb_frame.f_code.co_filename == PATH:
            detail["line"] = frame.tb_lineno
        frame = frame.tb_next
    if phase == "compile" and isinstance(exc, SyntaxError):
        detail.update(line=exc.lineno, column=exc.offset, reason=exc.msg)
    return sanitize_detail(detail)


def repair_detail(value):
    detail = sanitize_detail(value)
    if not detail:
        return detail
    if detail["phase"] == "Policy.schedule.result":
        hint = "return an acyclic JSON-compatible decision with finite numbers and supported value types"
    else:
        hint = HINTS.get(detail.get("exception_type"),
                         "repair the candidate exception at the reported phase and location")
    if detail.get("reason"):
        hint = detail["reason"]
    return {**detail, "hint": hint}
