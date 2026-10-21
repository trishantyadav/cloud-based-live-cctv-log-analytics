import argparse
import json
from pathlib import Path
from .runtime import Runtime


def main():
    parser = argparse.ArgumentParser(description='CCTV RAG ingestion/query CLI')
    parser.add_argument('action', choices=['ingest', 'ask', 'status', 'reindex'])
    parser.add_argument('value', nargs='?')
    parser.add_argument('--batch-id')
    parser.add_argument('--start')
    parser.add_argument('--end')
    parser.add_argument('--camera')
    parser.add_argument('--source', choices=['synthetic', 'live'])
    args = parser.parse_args()
    runtime = None
    try:
        runtime = Runtime()
        if args.action == 'ingest':
            if not args.value or not args.batch_id:
                parser.error('ingest requires a JSON file and --batch-id')
            result = runtime.ingestor.ingest_batch(json.loads(Path(args.value).read_text()), args.batch_id)
        elif args.action == 'ask':
            result = runtime.service.ask(args.value or '', start=args.start, end=args.end, camera=args.camera, source=args.source)
        elif args.action == 'reindex':
            with runtime.store.ingestion_lock():
                for summary in runtime.store.summaries():
                    with runtime.store.transaction():
                        runtime.processor.refresh(summary['camera_id'], summary['start'], summary['end'], summary['source_type'])
            result = {'status': 'reindexed', 'note': 'Explicit maintenance; normal ingestion only updates affected windows.'}
        else:
            result = runtime.status()
        print(json.dumps(result, indent=2))
        if result.get('status') in {'failed', 'partially_failed'}:
            raise SystemExit(1)
    except Exception as exc:
        # No raw DB/provider exceptions in console output.
        from .query_parser import Clarification
        print(str(exc) if isinstance(exc, Clarification) else 'Operation failed; check configuration, input and service availability.')
        raise SystemExit(1)
    finally:
        if runtime:
            runtime.close()


if __name__ == '__main__':
    main()
