# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
'''The CLI entrypoints of vLLM

Note that all future modules must be lazily loaded within main
to avoid certain eager import breakage.'''
from __future__ import annotations

import importlib.metadata


def main():
    print("[TRACE] (main.py) vLLM CLI main() called")
    import vllm.entrypoints.cli.benchmark.main
    import vllm.entrypoints.cli.collect_env
    import vllm.entrypoints.cli.openai
    import vllm.entrypoints.cli.run_batch
    import vllm.entrypoints.cli.serve
    from vllm.entrypoints.utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup
    from vllm.utils import FlexibleArgumentParser

    CMD_MODULES = [
        vllm.entrypoints.cli.openai,
        vllm.entrypoints.cli.serve,
        vllm.entrypoints.cli.benchmark.main,
        vllm.entrypoints.cli.collect_env,
        vllm.entrypoints.cli.run_batch,
    ]

    print("[TRACE] (main.py) Running cli_env_setup()")
    cli_env_setup()
    print("[TRACE] (main.py) cli_env_setup() completed")

    print("[TRACE] Creating argument parser")
    parser = FlexibleArgumentParser(
        description="vLLM CLI",
        epilog=VLLM_SUBCMD_PARSER_EPILOG,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=importlib.metadata.version('vllm'),
    )
    subparsers = parser.add_subparsers(required=False, dest="subparser")
    cmds = {}
    for cmd_module in CMD_MODULES:
        new_cmds = cmd_module.cmd_init()
        for cmd in new_cmds:
            cmd.subparser_init(subparsers).set_defaults(
                dispatch_function=cmd.cmd)
            cmds[cmd.name] = cmd
    args = parser.parse_args()
    print(f"[TRACE] (main.py) Arguments parsed, subparser: {args.subparser}")
    if args.subparser in cmds:
        print(f"[TRACE] Validating arguments for subcommand: {args.subparser}")
        cmds[args.subparser].validate(args)

    if hasattr(args, "dispatch_function"):
        print(f"[TRACE] (main.py) Dispatching to {args.subparser} command")
        args.dispatch_function(args)
    else:
        print("[TRACE] (main.py) No subcommand specified, printing help")
        parser.print_help()


if __name__ == "__main__":
    main()
