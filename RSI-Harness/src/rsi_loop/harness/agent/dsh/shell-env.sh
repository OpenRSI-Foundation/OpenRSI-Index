# BASH_ENV of the dsh Agent, whose shell tool runs every command with
# `bash -c` in an environment without the variables whose names hold KEY,
# PASSWORD, SECRET or TOKEN (rsi-submit's RSI_TOKEN among them). The launch
# kept their declarations in RSI_DSH_SHELL_ENV; restore them, then source the
# BASH_ENV the launch had.
if [ -n "${RSI_DSH_SHELL_ENV-}" ]; then
    eval "$RSI_DSH_SHELL_ENV" 2>/dev/null
fi
if [ -n "${RSI_DSH_BASH_ENV-}" ]; then
    . "$RSI_DSH_BASH_ENV"
fi
