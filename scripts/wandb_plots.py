import wandb
api = wandb.Api()


run = api.run("<entity>/<project>/<run_id>")