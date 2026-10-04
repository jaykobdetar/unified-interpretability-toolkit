"""Local owner interface; acquisition is explicit and has no HTTP action route."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from .common import read_json
from .config import capabilities, load_config
from .registry import Registry, content_digest, validate_manifest, MAX_MANIFEST_BYTES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('validate-config')
    planned = commands.add_parser('plan', help='Validate a supplied pinned manifest; no source or network reads')
    planned.add_argument('--manifest', required=True, type=Path)
    register = commands.add_parser('register', help='Hash already-installed bounded files; no acquisition')
    register.add_argument('--manifest', required=True, type=Path)
    register.add_argument('--source', required=True, type=Path)
    register.add_argument('--name', required=True)
    register.add_argument('--max-bytes', type=int, default=1024**2)
    register.add_argument('--timeout-ms', type=int, default=1000)
    commands.add_parser('list', help='Print visitor-safe local installed-model catalog')
    for command in ('enable', 'disable', 'receipt', 'prepare-fixture'):
        item = commands.add_parser(command)
        item.add_argument('model_id')
    commands.add_parser('download', help='Disabled until a reviewed acquisition stage')
    for command in ('acquire-plan', 'acquire'):
        item = commands.add_parser(command, help='Explicit owner pinned-data acquisition '+command)
        item.add_argument('--manifest', required=True, type=Path)
        item.add_argument('--name', required=True)
        item.add_argument('--max-bytes', required=True, type=int)
        item.add_argument('--cache-growth-bytes', type=int, default=0)
        if command == 'acquire':
            item.add_argument('--destination', required=True, type=Path)
            item.add_argument('--plan-digest', required=True)
            item.add_argument('--accept-license', action='store_true')
            item.add_argument('--timeout-ms', type=int, default=120000)
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        registry = Registry(config['paths']['registry'])
        if args.command == 'validate-config':
            result = capabilities(config)
        elif args.command == 'download':
            raise ValueError('Downloads are disabled; no acquisition authorized by this local stage')
        elif args.command in ('acquire-plan', 'acquire'):
            from .acquisition import plan, acquire
            manifest = read_json(args.manifest, MAX_MANIFEST_BYTES)
            options = {'max_bytes': args.max_bytes, 'cache_growth': args.cache_growth_bytes}
            if args.command == 'acquire-plan':
                result = plan(manifest, args.name, **options)
            else:
                result = acquire(registry, args.destination, manifest, args.name,
                                 plan_digest=args.plan_digest, accept_license=args.accept_license,
                                 timeout_ms=args.timeout_ms, **options)
                if not result['registered']:
                    print(json.dumps(result, sort_keys=True, allow_nan=False))
                    return 2
        elif args.command == 'plan':
            manifest = validate_manifest(read_json(args.manifest, MAX_MANIFEST_BYTES))
            result = {'version': 1, 'model_id': 'm_'+content_digest(manifest),
                      'repository': manifest['repository'], 'revision': manifest['revision'],
                      'files': manifest['files'], 'license': manifest['license'],
                      'bytes_to_verify': sum(file['bytes'] for file in manifest['files']),
                      'download_enabled': False, 'inference_ready': False}
        elif args.command == 'register':
            result = registry.register(args.source, read_json(args.manifest, MAX_MANIFEST_BYTES),
                                       args.name, max_bytes=args.max_bytes, timeout_ms=args.timeout_ms)
        elif args.command == 'list':
            result = registry.catalog()
        elif args.command == 'receipt':
            result = registry.owner_receipt(args.model_id)
        elif args.command == 'prepare-fixture':
            from .prepare_fixture import prepare
            result = prepare(config, args.model_id)
        else:
            result = registry.set_enabled(args.model_id, args.command == 'enable')
        print(json.dumps(result, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        # Owner-local stderr only; an HTTP adapter must emit fixed public codes.
        print('atlas-model: '+str(error), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
