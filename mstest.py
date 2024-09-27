954990 954991 954992 954993 954994 954995 954996
954990|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954991|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954992|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954993|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954994|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954995|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954996|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954997|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954998|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
954999|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955000|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955001|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955002|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955003|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955004|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955005|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955006|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955007|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955008|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955009|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955010|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955011|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955012|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0
955013|medium_uplot_embedding_start|plgrid-gpu-a100|plgllmefficont-gpu-a100|0|PENDING|0:0








946329       base_uplo+ plgrid-gp+ plgllmeff+         32     FAILED      1:0 


946330       base_uplo+ plgrid-gp+ plgllmeff+         32     FAILED      1:0 
edge case - błąd z zapisywaniem wykresu, błąd ścieżki


948438       base_uplo+ plgrid-gp+ plgllmeff+         32     FAILED      1:0
edge case - błąd z zapisywaniem wykresu, błąd ścieżki

Shutting down background jobs, please wait a moment...
Done!
Waiting for the remaining 1 operations to synchronize with Neptune. Do not kill this process.
All 1 operations synced, thanks for waiting!
Explore the metadata in the Neptune app:
https://app.neptune.ai/pmtest/llm-random/e/LLMRANDOM-15709/metadata
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/research/conditional/train/cc_train.py", line 494, in <module>
    mp.spawn(
  File "/opt/venv/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 246, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method="spawn")
  File "/opt/venv/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 202, in start_processes
    while not context.join():
  File "/opt/venv/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 163, in join
    raise ProcessRaisedException(msg, error_index, failed_process.pid)
torch.multiprocessing.spawn.ProcessRaisedException: 

-- Process 0 terminated with the following error:
OSError: [Errno 5] Input/output error

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/opt/venv/lib/python3.10/site-packages/torch/multiprocessing/spawn.py", line 74, in _wrap
    fn(i, *args)
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/research/conditional/train/cc_train.py", line 472, in main
    trainer.train(args.n_steps)
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/research/conditional/utils/conditional_trainer.py", line 148, in train
    self._train_step(step)
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/research/conditional/utils/conditional_trainer.py", line 185, in _train_step
    self.layer_manager.log(step)
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/research/conditional/utils/layer_manager.py", line 89, in log
    self.logger.report_generic_info(
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/lizrd/support/logging.py", line 416, in report_generic_info
    logger.report_generic_info(title=title, iteration=iteration, data=data)
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/lizrd/support/logging.py", line 205, in report_generic_info
    self.report_plotly(figure=data, title=title, iteration=iteration)
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/lizrd/support/logging.py", line 268, in report_plotly
    self._upload_with_tmp_file(f"{directory}/plot_{filename}", html, "html")
  File "/net/pr2/projects/plgrid/plggllmeffi/plgmstefaniak/llm_random_cemetery/base_uplot_embedding_start_2024-09-26_10-46-05/lizrd/support/logging.py", line 199, in _upload_with_tmp_file
    with open(tmp_file, "w") as f:
OSError: [Errno 5] Input/output error