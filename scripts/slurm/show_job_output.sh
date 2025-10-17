# Takes as an argument a job_id: prints the job output
tail -f $(scontrol show job $1 | awk -F= '/StdOut/ {print $2}')
