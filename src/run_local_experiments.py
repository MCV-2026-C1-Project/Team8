"""Run a validated Week 4 development grid with reusable exhaustive evidence."""
import argparse
import json
from pathlib import Path
from local_matching_config import ROOT
from local_experiment_config import expand_local_experiments
from local_experiments import run_local_experiments
from local_experiment_output import write_outputs, compare_week3


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--query-dir",type=Path,default=ROOT/"data/qsd1_w4")
    parser.add_argument("--db-dir",type=Path,default=ROOT/"data/BBDD")
    parser.add_argument("--cache-dir",type=Path,default=ROOT/"descriptors")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--query-limit",type=int)
    parser.add_argument("--experiment-limit",type=int)
    parser.add_argument("--week3-config",type=Path)
    parser.add_argument("--diagnostics",action="store_true")
    args=parser.parse_args(argv)
    try:
        for name,value in (("query-limit",args.query_limit),("experiment-limit",args.experiment_limit)):
            if value is not None and value < 1: raise ValueError(f"{name} must be positive")
        jobs=expand_local_experiments(json.loads(args.config.read_text(encoding="utf-8")),args.config.resolve().parent)
        jobs=jobs[:args.experiment_limit]
        print(f"{len(jobs)} configurations; exhaustive database retrieval")
        if args.dry_run:
            for job in jobs: print(json.dumps(job,sort_keys=True))
            return jobs
        if args.output is None: raise ValueError("--output is required unless --dry-run")
        args.output.mkdir(parents=True,exist_ok=False)
        run=run_local_experiments(jobs,args.query_dir,args.db_dir,args.cache_dir,query_limit=args.query_limit)
        baseline=compare_week3(args.week3_config,args.query_dir,args.db_dir,args.cache_dir,
                               json.loads((ROOT/"configs/week3/task3_selected.json").read_text(encoding="utf-8")),
                               query_limit=args.query_limit,
                               k=max(j["config"]["retrieval"]["k"] for j in jobs))
        summary=write_outputs(args.output,run,args.query_dir,args.db_dir,baseline,args.diagnostics)
        print(f"New stage computations (shared across configurations): {run['counts']}")
        print(f"Completed: best={summary['best_id']}; outputs={args.output.resolve()}")
        for warning in summary["warnings"]: print(f"Warning: {warning}")
        return summary
    except (ValueError,OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
